/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFACUDA_CUDAPCGLINEARSOLVER_CPP

#include <SofaCUDA/component/config.h>

#ifdef SOFA_GPU_CUBLAS

#include <SofaCUDA/component/linearsolver/iterative/CudaPCGLinearSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/gpu/cuda/mycuda.h>

#include <cmath>
#include <algorithm>

namespace sofa::gpu::cuda
{

template<class TMatrix, class TVector>
CudaPCGLinearSolver<TMatrix, TVector>::CudaPCGLinearSolver()
    : d_maxIter(initData(&d_maxIter, 25u, "iterations",
        "Maximum number of iterations after which the iterative descent of the Conjugate Gradient must stop"))
    , d_tolerance(initData(&d_tolerance, (Real)1e-5, "tolerance",
        "Desired accuracy of the Conjugate Gradient solution evaluating: |r|^2/|b|^2"))
    , d_smallDenominatorThreshold(initData(&d_smallDenominatorThreshold, (Real)1e-5, "threshold",
        "Minimum value of the denominator (pT A p) in the conjugate Gradient solution"))
    , d_warmStart(initData(&d_warmStart, false, "warmStart",
        "Use previous solution as initial solution"))
    , d_usePreconditioner(initData(&d_usePreconditioner, true, "usePreconditioner",
        "Use the linked preconditioner"))
    , d_graph(initData(&d_graph, "graph", "Graph of residuals at each iteration"))
    , l_preconditioner(initLink("preconditioner",
        "Link towards the linear solver used to precondition the conjugate gradient"))
{
    d_graph.setWidget("graph");
    d_maxIter.setRequired(true);
    d_tolerance.setRequired(true);
    d_smallDenominatorThreshold.setRequired(true);
}

template<class TMatrix, class TVector>
CudaPCGLinearSolver<TMatrix, TVector>::~CudaPCGLinearSolver()
{
    freeCudaResources();
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::init()
{
    Inherit::init();

    if (d_tolerance.getValue() < 0.0)
    {
        msg_warning() << "'tolerance' must be a positive value, using default: 1e-5";
        d_tolerance.setValue(1e-5);
    }
    if (d_smallDenominatorThreshold.getValue() < 0.0)
    {
        msg_warning() << "'threshold' must be a positive value, using default: 1e-5";
        d_smallDenominatorThreshold.setValue(1e-5);
    }

    if (l_preconditioner.empty())
    {
        msg_info() << "No preconditioner linked. The solver will behave as a standard CG solver. "
                   << "Link a CUDA preconditioner (e.g., CudaJacobiPreconditioner, CudaIC0Preconditioner) "
                   << "via the 'preconditioner' link for better convergence.";
    }
    else if (l_preconditioner.get() == nullptr)
    {
        msg_error() << "No preconditioner found at path: " << l_preconditioner.getLinkedPath();
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    else
    {
        auto* cudaPrecond = dynamic_cast<CudaPreconditionerBase*>(l_preconditioner.get());
        if (cudaPrecond)
        {
            msg_info() << "Using CUDA preconditioner: " << l_preconditioner->getName();
        }
        else
        {
            msg_warning() << "Linked preconditioner '" << l_preconditioner->getName()
                          << "' is not a CUDA preconditioner. It will be ignored. "
                          << "Use CudaJacobiPreconditioner or CudaIC0Preconditioner for GPU acceleration.";
        }
    }

    m_timeStepCount = 0;
    m_equilibriumReached = false;

    mycudaInit();
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::allocateCudaResources(int nRows, int nCols, int nnz)
{
    if (m_cudaData.allocated &&
        m_cudaData.nRows == nRows &&
        m_cudaData.nCols == nCols &&
        m_cudaData.nnz == nnz)
    {
        return;
    }

    freeCudaResources();

    m_cudaData.nRows = nRows;
    m_cudaData.nCols = nCols;
    m_cudaData.nnz = nnz;

    // Allocate CSR matrix data
    mycudaMalloc(&m_cudaData.d_csrVal, nnz * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_csrRowPtr, (nRows + 1) * sizeof(int));
    mycudaMalloc(&m_cudaData.d_csrColInd, nnz * sizeof(int));

    // Allocate vectors
    mycudaMalloc(&m_cudaData.d_x, nCols * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_b, nRows * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_r, nRows * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_z, nRows * sizeof(Real));  // preconditioned residual
    mycudaMalloc(&m_cudaData.d_p, nCols * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_Ap, nRows * sizeof(Real));

    // Create cuSPARSE matrix descriptor
    cudaDataType valueType = (sizeof(Real) == sizeof(float)) ? CUDA_R_32F : CUDA_R_64F;

    cusparseCreateCsr(&m_cudaData.matA, nRows, nCols, nnz,
                      m_cudaData.d_csrRowPtr, m_cudaData.d_csrColInd, m_cudaData.d_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, valueType);

    // Create dense vector descriptors
    cusparseCreateDnVec(&m_cudaData.vecX, nCols, m_cudaData.d_p, valueType);
    cusparseCreateDnVec(&m_cudaData.vecY, nRows, m_cudaData.d_Ap, valueType);

    m_cudaData.allocated = true;
    m_cudaData.bufferSize = 0;  // Will be computed after matrix data is uploaded
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::freeCudaResources()
{
    if (!m_cudaData.allocated) return;

    if (m_cudaData.matA) cusparseDestroySpMat(m_cudaData.matA);
    if (m_cudaData.vecX) cusparseDestroyDnVec(m_cudaData.vecX);
    if (m_cudaData.vecY) cusparseDestroyDnVec(m_cudaData.vecY);

    if (m_cudaData.d_csrVal) mycudaFree(m_cudaData.d_csrVal);
    if (m_cudaData.d_csrRowPtr) mycudaFree(m_cudaData.d_csrRowPtr);
    if (m_cudaData.d_csrColInd) mycudaFree(m_cudaData.d_csrColInd);
    if (m_cudaData.d_x) mycudaFree(m_cudaData.d_x);
    if (m_cudaData.d_b) mycudaFree(m_cudaData.d_b);
    if (m_cudaData.d_r) mycudaFree(m_cudaData.d_r);
    if (m_cudaData.d_z) mycudaFree(m_cudaData.d_z);
    if (m_cudaData.d_p) mycudaFree(m_cudaData.d_p);
    if (m_cudaData.d_Ap) mycudaFree(m_cudaData.d_Ap);
    if (m_cudaData.d_buffer) mycudaFree(m_cudaData.d_buffer);

    m_cudaData = CudaSparseData();
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::uploadMatrix(const Matrix& A)
{
    const auto& rowBegin = A.getRowBegin();
    const auto& colsIndex = A.getColsIndex();
    const auto& colsValue = A.getColsValue();

    using Block = typename Matrix::Block;
    constexpr int BlockRows = Matrix::NL;
    constexpr int BlockCols = Matrix::NC;

    const int blockRows = static_cast<int>(A.nBlockRow);
    const int blockCols = static_cast<int>(A.nBlockCol);
    const int scalarRows = blockRows * BlockRows;
    const int scalarCols = blockCols * BlockCols;

    // Build scalar CSR from block CSR
    std::vector<Real> values;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;

    rowPtr.reserve(scalarRows + 1);
    rowPtr.push_back(0);

    for (int br = 0; br < blockRows; ++br)
    {
        for (int lr = 0; lr < BlockRows; ++lr)
        {
            // Collect all entries for this scalar row
            std::vector<std::pair<int, Real>> rowEntries;

            for (auto idx = rowBegin[br]; idx < rowBegin[br + 1]; ++idx)
            {
                const int bc = static_cast<int>(colsIndex[idx]);
                const Block& block = colsValue[idx];

                for (int lc = 0; lc < BlockCols; ++lc)
                {
                    const int scalarCol = bc * BlockCols + lc;
                    Real val;
                    if constexpr (BlockRows == 1 && BlockCols == 1)
                    {
                        val = static_cast<Real>(block);
                    }
                    else
                    {
                        val = static_cast<Real>(block(lr, lc));
                    }

                    if (val != Real(0))
                    {
                        rowEntries.emplace_back(scalarCol, val);
                    }
                }
            }

            // Sort by column index (required for CSR format by cuSPARSE)
            std::sort(rowEntries.begin(), rowEntries.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

            for (const auto& entry : rowEntries)
            {
                colIdx.push_back(entry.first);
                values.push_back(entry.second);
            }

            rowPtr.push_back(static_cast<int>(values.size()));
        }
    }

    const int nnz = static_cast<int>(values.size());

    const bool needsRealloc = !m_cudaData.allocated ||
                              m_cudaData.nRows != scalarRows ||
                              m_cudaData.nCols != scalarCols ||
                              m_cudaData.nnz != nnz;

    allocateCudaResources(scalarRows, scalarCols, nnz);

    mycudaMemcpyHostToDevice(m_cudaData.d_csrVal, values.data(), nnz * sizeof(Real));
    mycudaMemcpyHostToDevice(m_cudaData.d_csrRowPtr, rowPtr.data(), (scalarRows + 1) * sizeof(int));
    mycudaMemcpyHostToDevice(m_cudaData.d_csrColInd, colIdx.data(), nnz * sizeof(int));

    cusparseCsrSetPointers(m_cudaData.matA,
                           m_cudaData.d_csrRowPtr,
                           m_cudaData.d_csrColInd,
                           m_cudaData.d_csrVal);

    // Compute SpMV buffer size after matrix data is uploaded (only on reallocation)
    if (needsRealloc)
    {
        cudaDataType valueType = (sizeof(Real) == sizeof(float)) ? CUDA_R_32F : CUDA_R_64F;
        Real alpha = 1.0;
        Real beta = 0.0;

        if (m_cudaData.d_buffer)
        {
            mycudaFree(m_cudaData.d_buffer);
            m_cudaData.d_buffer = nullptr;
        }

        cusparseSpMV_bufferSize(getCusparseCtx(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, m_cudaData.matA, m_cudaData.vecX, &beta, m_cudaData.vecY,
                                valueType, CUSPARSE_SPMV_ALG_DEFAULT, &m_cudaData.bufferSize);

        if (m_cudaData.bufferSize > 0)
        {
            mycudaMalloc(&m_cudaData.d_buffer, m_cudaData.bufferSize);
        }
    }
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::uploadVector(const Vector& src, void* d_dst)
{
    mycudaMemcpyHostToDevice(d_dst, src.ptr(), src.size() * sizeof(Real));
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::downloadVector(void* d_src, Vector& dst)
{
    mycudaMemcpyDeviceToHost(dst.ptr(), d_src, dst.size() * sizeof(Real));
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::sparseMatVec(void* d_x, void* d_y)
{
    Real alpha = 1.0;
    Real beta = 0.0;
    cudaDataType valueType = (sizeof(Real) == sizeof(float)) ? CUDA_R_32F : CUDA_R_64F;

    cusparseDnVecSetValues(m_cudaData.vecX, d_x);
    cusparseDnVecSetValues(m_cudaData.vecY, d_y);

    cusparseStatus_t status = cusparseSpMV(getCusparseCtx(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, m_cudaData.matA, m_cudaData.vecX, &beta, m_cudaData.vecY,
                 valueType, CUSPARSE_SPMV_ALG_DEFAULT, m_cudaData.d_buffer);

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        msg_error() << "cusparseSpMV failed with status: " << status;
    }
}

template<class TMatrix, class TVector>
typename CudaPCGLinearSolver<TMatrix, TVector>::Real
CudaPCGLinearSolver<TMatrix, TVector>::gpuDot(void* d_a, void* d_b, int n)
{
    Real result = 0.0;
    cublasHandle_t handle = getCublasCtx();
    cublasStatus_t status;

    if constexpr (sizeof(Real) == sizeof(float))
    {
        float fresult = 0.0f;
        status = cublasSdot(handle, n, static_cast<const float*>(d_a), 1,
                   static_cast<const float*>(d_b), 1, &fresult);
        result = static_cast<Real>(fresult);
    }
    else
    {
        status = cublasDdot(handle, n, static_cast<const double*>(d_a), 1,
                   static_cast<const double*>(d_b), 1, &result);
    }

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        msg_error() << "cublasDot failed with status: " << status;
    }

    return result;
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::gpuAxpy(Real alpha, void* d_x, void* d_y, int n)
{
    cublasHandle_t handle = getCublasCtx();

    if constexpr (sizeof(Real) == sizeof(float))
    {
        float a = static_cast<float>(alpha);
        cublasSaxpy(handle, n, &a, static_cast<const float*>(d_x), 1,
                    static_cast<float*>(d_y), 1);
    }
    else
    {
        cublasDaxpy(handle, n, &alpha, static_cast<const double*>(d_x), 1,
                    static_cast<double*>(d_y), 1);
    }
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::gpuCopy(void* d_src, void* d_dst, int n)
{
    cublasHandle_t handle = getCublasCtx();

    if constexpr (sizeof(Real) == sizeof(float))
    {
        cublasScopy(handle, n, static_cast<const float*>(d_src), 1,
                    static_cast<float*>(d_dst), 1);
    }
    else
    {
        cublasDcopy(handle, n, static_cast<const double*>(d_src), 1,
                    static_cast<double*>(d_dst), 1);
    }
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::gpuScale(Real alpha, void* d_x, int n)
{
    cublasHandle_t handle = getCublasCtx();

    if constexpr (sizeof(Real) == sizeof(float))
    {
        float a = static_cast<float>(alpha);
        cublasSscal(handle, n, &a, static_cast<float*>(d_x), 1);
    }
    else
    {
        cublasDscal(handle, n, &alpha, static_cast<double*>(d_x), 1);
    }
}

template<class TMatrix, class TVector>
void CudaPCGLinearSolver<TMatrix, TVector>::solve(Matrix& A, Vector& x, Vector& b)
{
    sofa::helper::AdvancedTimer::stepBegin("CudaPCG-Solve");

    const int n = static_cast<int>(b.size());

    msg_info() << "CudaPCG::solve called, n=" << n;

    // Upload matrix to GPU
    uploadMatrix(A);

    // Check if preconditioner is available and should be used
    const bool usePrecond = d_usePreconditioner.getValue() && l_preconditioner.get() != nullptr;

    // Check if preconditioner supports GPU-native solve (no CPU-GPU transfers)
    CudaPreconditionerBase* cudaPrecond = nullptr;
    if (usePrecond)
    {
        cudaPrecond = dynamic_cast<CudaPreconditionerBase*>(l_preconditioner.get());
    }

    if (usePrecond)
    {
        // Build preconditioner
        sofa::helper::AdvancedTimer::stepBegin("CudaPCG-BuildPreconditioner");
        if (cudaPrecond)
        {
            // Use the CUDA preconditioner interface
            cudaPrecond->updatePreconditioner(&A);
        }
        sofa::helper::AdvancedTimer::stepEnd("CudaPCG-BuildPreconditioner");
    }

    const bool useGPUPrecond = cudaPrecond != nullptr && cudaPrecond->isReadyForGPU();


    // Initialize graph for residuals
    m_newtonIter++;
    std::map<std::string, sofa::type::vector<Real>>& graph = *d_graph.beginEdit();
    char name[256];
    sprintf(name, "Error %d", m_newtonIter);
    sofa::type::vector<Real>& graph_error = graph[std::string(name)];
    graph_error.clear();

    // Upload vectors to GPU
    uploadVector(b, m_cudaData.d_b);

    Real rho, rho_1 = 0, alpha, beta;
    unsigned nb_iter = 0;
    const char* endcond = "iterations";

    // Compute initial residual
    if (d_warmStart.getValue())
    {
        uploadVector(x, m_cudaData.d_x);
        sparseMatVec(m_cudaData.d_x, m_cudaData.d_r);
        // r = b - A*x
        gpuScale(Real(-1.0), m_cudaData.d_r, n);
        gpuAxpy(Real(1.0), m_cudaData.d_b, m_cudaData.d_r, n);
    }
    else
    {
        mycudaMemset(m_cudaData.d_x, 0, n * sizeof(Real));
        gpuCopy(m_cudaData.d_b, m_cudaData.d_r, n);
    }

    // Compute norm of b
    const Real normb = std::sqrt(gpuDot(m_cudaData.d_b, m_cudaData.d_b, n));

    graph_error.push_back(Real(1.0));

    if (normb != 0.0)
    {
        // Initial preconditioner application: z = M^{-1} * r
        if (useGPUPrecond)
        {
            cudaPrecond->solveOnGPU(m_cudaData.d_z, m_cudaData.d_r, n);
        }
        else
        {
            gpuCopy(m_cudaData.d_r, m_cudaData.d_z, n);
        }

        // Initial rho = r^T * z
        rho = gpuDot(m_cudaData.d_r, m_cudaData.d_z, n);

        // p = z
        gpuCopy(m_cudaData.d_z, m_cudaData.d_p, n);

        for (nb_iter = 1; nb_iter <= d_maxIter.getValue(); ++nb_iter)
        {
            // Compute error = |r| / |b|
            const Real normr = std::sqrt(gpuDot(m_cudaData.d_r, m_cudaData.d_r, n));
            const Real err = normr / normb;
            graph_error.push_back(err);

            // Check tolerance: |r|/|b| <= tolerance
            if (err <= d_tolerance.getValue())
            {
                if (nb_iter == 1 && m_timeStepCount == 0)
                {
                    msg_warning() << "tolerance reached at first iteration of PCG";
                }
                else
                {
                    if (nb_iter == 1 && !m_equilibriumReached)
                    {
                        msg_info() << "Equilibrium reached regarding tolerance";
                        m_equilibriumReached = true;
                    }
                    if (nb_iter > 1)
                    {
                        m_equilibriumReached = false;
                    }
                    endcond = "tolerance";
                    msg_info() << "error = " << err << ", tolerance = " << d_tolerance.getValue();
                    break;
                }
            }

            // Compute Ap = A * p
            sparseMatVec(m_cudaData.d_p, m_cudaData.d_Ap);

            // Compute denominator: den = p^T * Ap
            const Real den = gpuDot(m_cudaData.d_p, m_cudaData.d_Ap, n);

            if (den == 0.0)
            {
                msg_warning() << "den = 0.0, breaking iterations";
                break;
            }

            if (std::fabs(den) <= d_smallDenominatorThreshold.getValue())
            {
                if (nb_iter == 1 && m_timeStepCount == 0)
                {
                    msg_warning() << "denominator threshold reached at first iteration of PCG";
                }
                else
                {
                    endcond = "threshold";
                    msg_info() << "den = " << den << ", threshold = " << d_smallDenominatorThreshold.getValue();
                    break;
                }
            }

            // alpha = rho / den
            alpha = rho / den;

            // x = x + alpha * p
            gpuAxpy(alpha, m_cudaData.d_p, m_cudaData.d_x, n);

            // r = r - alpha * Ap
            gpuAxpy(-alpha, m_cudaData.d_Ap, m_cudaData.d_r, n);

            // z = M^{-1} * r
            if (useGPUPrecond)
            {
                cudaPrecond->solveOnGPU(m_cudaData.d_z, m_cudaData.d_r, n);
            }
            else
            {
                gpuCopy(m_cudaData.d_r, m_cudaData.d_z, n);
            }

            // rho_new = r^T * z
            rho_1 = rho;
            rho = gpuDot(m_cudaData.d_r, m_cudaData.d_z, n);

            // beta = rho_new / rho_old
            beta = rho / rho_1;

            // p = z + beta * p
            gpuScale(beta, m_cudaData.d_p, n);
            gpuAxpy(Real(1.0), m_cudaData.d_z, m_cudaData.d_p, n);
        }
    }
    else
    {
        endcond = "null norm of vector b";
    }

    // Download solution from GPU
    downloadVector(m_cudaData.d_x, x);

    d_graph.endEdit();
    m_timeStepCount++;

    sofa::helper::AdvancedTimer::valSet("CudaPCG iterations", nb_iter);
    sofa::helper::AdvancedTimer::stepEnd("CudaPCG-Solve");

    msg_info() << "solve, nbiter = " << nb_iter << " stop because of " << endcond;
    msg_info() << "solve, solution = " << x;
}

// Explicit template instantiations
template class SOFACUDA_COMPONENT_API CudaPCGLinearSolver<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;

template class SOFACUDA_COMPONENT_API CudaPCGLinearSolver<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;

// Component registration
void registerCudaPCGLinearSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        sofa::core::ObjectRegistrationData(
            "CUDA-accelerated Preconditioned Conjugate Gradient linear solver. "
            "Uses cuSPARSE for SpMV and cuBLAS for vector operations. "
            "Link a preconditioner (CudaJacobiPreconditioner or CudaIC0Preconditioner) for better convergence.")
        .add<CudaPCGLinearSolver<linearalgebra::CompressedRowSparseMatrix<SReal>,
                                  linearalgebra::FullVector<SReal>>>()
        .add<CudaPCGLinearSolver<linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
                                  linearalgebra::FullVector<SReal>>>()
    );
}

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
