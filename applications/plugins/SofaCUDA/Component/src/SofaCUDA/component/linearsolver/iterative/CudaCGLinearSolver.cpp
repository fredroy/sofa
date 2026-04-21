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
#define SOFACUDA_CUDACGLINEARSOLVER_CPP

#include <SofaCUDA/component/config.h>

#ifdef SOFA_GPU_CUBLAS

#include <SofaCUDA/component/linearsolver/iterative/CudaCGLinearSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/gpu/cuda/mycuda.h>

#include <cmath>
#include <algorithm>

namespace sofa::gpu::cuda
{

template<class TMatrix, class TVector>
CudaCGLinearSolver<TMatrix, TVector>::CudaCGLinearSolver()
    : d_maxIter(initData(&d_maxIter, 25u, "iterations",
        "Maximum number of iterations after which the iterative descent of the Conjugate Gradient must stop"))
    , d_tolerance(initData(&d_tolerance, (Real)1e-5, "tolerance",
        "Desired accuracy of the Conjugate Gradient solution evaluating: |r|^2/|b|^2"))
    , d_smallDenominatorThreshold(initData(&d_smallDenominatorThreshold, (Real)1e-5, "threshold",
        "Minimum value of the denominator (pT A p) in the conjugate Gradient solution"))
    , d_warmStart(initData(&d_warmStart, false, "warmStart",
        "Use previous solution as initial solution"))
    , d_graph(initData(&d_graph, "graph", "Graph of residuals at each iteration"))
{
    d_graph.setWidget("graph");
    d_maxIter.setRequired(true);
    d_tolerance.setRequired(true);
    d_smallDenominatorThreshold.setRequired(true);
}

template<class TMatrix, class TVector>
CudaCGLinearSolver<TMatrix, TVector>::~CudaCGLinearSolver()
{
    freeCudaResources();
}

template<class TMatrix, class TVector>
void CudaCGLinearSolver<TMatrix, TVector>::init()
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

    m_timeStepCount = 0;
    m_equilibriumReached = false;

    // Initialize CUDA
    mycudaInit();
}

template<class TMatrix, class TVector>
void CudaCGLinearSolver<TMatrix, TVector>::allocateCudaResourcesCSR(int nRows, int nCols, int nnz)
{
    if (m_cudaData.allocated &&
        !m_cudaData.useBSR &&
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
    m_cudaData.blockDim = 1;
    m_cudaData.useBSR = false;

    // Allocate CSR matrix data
    mycudaMalloc(&m_cudaData.d_bsrVal, nnz * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_bsrRowPtr, (nRows + 1) * sizeof(int));
    mycudaMalloc(&m_cudaData.d_bsrColInd, nnz * sizeof(int));

    // Allocate vectors
    mycudaMalloc(&m_cudaData.d_x, nCols * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_b, nRows * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_r, nRows * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_p, nCols * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_Ap, nRows * sizeof(Real));

    // Create cuSPARSE CSR matrix descriptor
    cudaDataType valueType = (sizeof(Real) == sizeof(float)) ? CUDA_R_32F : CUDA_R_64F;

    cusparseCreateCsr(&m_cudaData.matA, nRows, nCols, nnz,
                      m_cudaData.d_bsrRowPtr, m_cudaData.d_bsrColInd, m_cudaData.d_bsrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, valueType);

    // Create dense vector descriptors
    cusparseCreateDnVec(&m_cudaData.vecX, nCols, m_cudaData.d_p, valueType);
    cusparseCreateDnVec(&m_cudaData.vecY, nRows, m_cudaData.d_Ap, valueType);

    // Determine buffer size for SpMV
    Real alpha = 1.0;
    Real beta = 0.0;
    cusparseSpMV_bufferSize(getCusparseCtx(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, m_cudaData.matA, m_cudaData.vecX, &beta, m_cudaData.vecY,
                            valueType, CUSPARSE_SPMV_ALG_DEFAULT, &m_cudaData.bufferSize);

    if (m_cudaData.bufferSize > 0)
    {
        mycudaMalloc(&m_cudaData.d_buffer, m_cudaData.bufferSize);
    }

    m_cudaData.allocated = true;
}

template<class TMatrix, class TVector>
void CudaCGLinearSolver<TMatrix, TVector>::allocateCudaResourcesBSR(int nBlockRows, int nBlockCols, int nnzBlocks, int blockDim)
{
    const int nRows = nBlockRows * blockDim;
    const int nCols = nBlockCols * blockDim;
    const int nnzValues = nnzBlocks * blockDim * blockDim;

    if (m_cudaData.allocated &&
        m_cudaData.useBSR &&
        m_cudaData.nBlockRows == nBlockRows &&
        m_cudaData.nBlockCols == nBlockCols &&
        m_cudaData.nnz == nnzBlocks &&
        m_cudaData.blockDim == blockDim)
    {
        return;
    }

    freeCudaResources();

    m_cudaData.nRows = nRows;
    m_cudaData.nCols = nCols;
    m_cudaData.nBlockRows = nBlockRows;
    m_cudaData.nBlockCols = nBlockCols;
    m_cudaData.nnz = nnzBlocks;
    m_cudaData.blockDim = blockDim;
    m_cudaData.useBSR = true;

    // Allocate BSR matrix data
    mycudaMalloc(&m_cudaData.d_bsrVal, nnzValues * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_bsrRowPtr, (nBlockRows + 1) * sizeof(int));
    mycudaMalloc(&m_cudaData.d_bsrColInd, nnzBlocks * sizeof(int));

    // Allocate vectors (scalar size)
    mycudaMalloc(&m_cudaData.d_x, nCols * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_b, nRows * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_r, nRows * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_p, nCols * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_Ap, nRows * sizeof(Real));

    // Create cuSPARSE BSR matrix descriptor
    cudaDataType valueType = (sizeof(Real) == sizeof(float)) ? CUDA_R_32F : CUDA_R_64F;

    cusparseCreateBsr(&m_cudaData.matA, nBlockRows, nBlockCols, nnzBlocks,
                      blockDim, blockDim,
                      m_cudaData.d_bsrRowPtr, m_cudaData.d_bsrColInd, m_cudaData.d_bsrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, valueType, CUSPARSE_ORDER_ROW);

    // Create dense vector descriptors (scalar size)
    cusparseCreateDnVec(&m_cudaData.vecX, nCols, m_cudaData.d_p, valueType);
    cusparseCreateDnVec(&m_cudaData.vecY, nRows, m_cudaData.d_Ap, valueType);

    // Determine buffer size for SpMV
    Real alpha = 1.0;
    Real beta = 0.0;
    cusparseSpMV_bufferSize(getCusparseCtx(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, m_cudaData.matA, m_cudaData.vecX, &beta, m_cudaData.vecY,
                            valueType, CUSPARSE_SPMV_ALG_DEFAULT, &m_cudaData.bufferSize);

    if (m_cudaData.bufferSize > 0)
    {
        mycudaMalloc(&m_cudaData.d_buffer, m_cudaData.bufferSize);
    }

    m_cudaData.allocated = true;

    msg_info() << "Using BSR format: " << nBlockRows << "x" << nBlockCols
               << " blocks of " << blockDim << "x" << blockDim
               << ", " << nnzBlocks << " non-zero blocks";
}

template<class TMatrix, class TVector>
void CudaCGLinearSolver<TMatrix, TVector>::freeCudaResources()
{
    if (!m_cudaData.allocated) return;

    if (m_cudaData.matA) cusparseDestroySpMat(m_cudaData.matA);
    if (m_cudaData.vecX) cusparseDestroyDnVec(m_cudaData.vecX);
    if (m_cudaData.vecY) cusparseDestroyDnVec(m_cudaData.vecY);

    if (m_cudaData.d_bsrVal) mycudaFree(m_cudaData.d_bsrVal);
    if (m_cudaData.d_bsrRowPtr) mycudaFree(m_cudaData.d_bsrRowPtr);
    if (m_cudaData.d_bsrColInd) mycudaFree(m_cudaData.d_bsrColInd);
    if (m_cudaData.d_x) mycudaFree(m_cudaData.d_x);
    if (m_cudaData.d_b) mycudaFree(m_cudaData.d_b);
    if (m_cudaData.d_r) mycudaFree(m_cudaData.d_r);
    if (m_cudaData.d_p) mycudaFree(m_cudaData.d_p);
    if (m_cudaData.d_Ap) mycudaFree(m_cudaData.d_Ap);
    if (m_cudaData.d_buffer) mycudaFree(m_cudaData.d_buffer);

    m_cudaData = CudaSparseData();
}

template<class TMatrix, class TVector>
void CudaCGLinearSolver<TMatrix, TVector>::uploadMatrix(const Matrix& A)
{
    const auto& rowBegin = A.getRowBegin();
    const auto& colsIndex = A.getColsIndex();
    const auto& colsValue = A.getColsValue();

    using Block = typename Matrix::Block;
    constexpr int BlockRows = Matrix::NL;
    constexpr int BlockCols = Matrix::NC;

    const int blockRows = static_cast<int>(A.nBlockRow);
    const int blockCols = static_cast<int>(A.nBlockCol);

    // Use BSR format for 3x3 blocks (common in Vec3 FEM)
    constexpr bool useBSR = (BlockRows == 3 && BlockCols == 3);

    if constexpr (useBSR)
    {
        // BSR format: keep blocks as-is, just need row pointers and column indices
        std::vector<Real> bsrValues;
        std::vector<int> bsrRowPtr;
        std::vector<int> bsrColIdx;

        bsrRowPtr.reserve(blockRows + 1);
        bsrRowPtr.push_back(0);

        for (int br = 0; br < blockRows; ++br)
        {
            // Collect blocks for this row, sorted by column
            std::vector<std::pair<int, const Block*>> rowBlocks;

            for (auto idx = rowBegin[br]; idx < rowBegin[br + 1]; ++idx)
            {
                const int bc = static_cast<int>(colsIndex[idx]);
                rowBlocks.emplace_back(bc, &colsValue[idx]);
            }

            // Sort by column index
            std::sort(rowBlocks.begin(), rowBlocks.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

            for (const auto& [bc, blockPtr] : rowBlocks)
            {
                bsrColIdx.push_back(bc);

                // Store block values in row-major order
                const Block& block = *blockPtr;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        bsrValues.push_back(static_cast<Real>(block(i, j)));
            }

            bsrRowPtr.push_back(static_cast<int>(bsrColIdx.size()));
        }

        const int nnzBlocks = static_cast<int>(bsrColIdx.size());

        // Allocate GPU resources for BSR
        allocateCudaResourcesBSR(blockRows, blockCols, nnzBlocks, 3);

        // Upload to GPU
        mycudaMemcpyHostToDevice(m_cudaData.d_bsrVal, bsrValues.data(), bsrValues.size() * sizeof(Real));
        mycudaMemcpyHostToDevice(m_cudaData.d_bsrRowPtr, bsrRowPtr.data(), (blockRows + 1) * sizeof(int));
        mycudaMemcpyHostToDevice(m_cudaData.d_bsrColInd, bsrColIdx.data(), nnzBlocks * sizeof(int));

        // For BSR, we need to update the matrix values directly
        // The pointers are set during allocation, values are updated via memcpy above
    }
    else
    {
        // CSR format for scalar or other block sizes
        const int scalarRows = blockRows * BlockRows;
        const int scalarCols = blockCols * BlockCols;

        std::vector<Real> values;
        std::vector<int> rowPtr;
        std::vector<int> colIdx;

        rowPtr.reserve(scalarRows + 1);
        rowPtr.push_back(0);

        for (int br = 0; br < blockRows; ++br)
        {
            for (int lr = 0; lr < BlockRows; ++lr)
            {
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

        allocateCudaResourcesCSR(scalarRows, scalarCols, nnz);

        mycudaMemcpyHostToDevice(m_cudaData.d_bsrVal, values.data(), nnz * sizeof(Real));
        mycudaMemcpyHostToDevice(m_cudaData.d_bsrRowPtr, rowPtr.data(), (scalarRows + 1) * sizeof(int));
        mycudaMemcpyHostToDevice(m_cudaData.d_bsrColInd, colIdx.data(), nnz * sizeof(int));

        cusparseCsrSetPointers(m_cudaData.matA,
                               m_cudaData.d_bsrRowPtr,
                               m_cudaData.d_bsrColInd,
                               m_cudaData.d_bsrVal);
    }
}

template<class TMatrix, class TVector>
void CudaCGLinearSolver<TMatrix, TVector>::uploadVector(const Vector& src, void* d_dst)
{
    mycudaMemcpyHostToDevice(d_dst, src.ptr(), src.size() * sizeof(Real));
}

template<class TMatrix, class TVector>
void CudaCGLinearSolver<TMatrix, TVector>::downloadVector(void* d_src, Vector& dst)
{
    mycudaMemcpyDeviceToHost(dst.ptr(), d_src, dst.size() * sizeof(Real));
}

template<class TMatrix, class TVector>
void CudaCGLinearSolver<TMatrix, TVector>::sparseMatVec(void* d_x, void* d_y)
{
    Real alpha = 1.0;
    Real beta = 0.0;
    cudaDataType valueType = (sizeof(Real) == sizeof(float)) ? CUDA_R_32F : CUDA_R_64F;

    // Update vector descriptors with current pointers
    cusparseDnVecSetValues(m_cudaData.vecX, d_x);
    cusparseDnVecSetValues(m_cudaData.vecY, d_y);

    cusparseSpMV(getCusparseCtx(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, m_cudaData.matA, m_cudaData.vecX, &beta, m_cudaData.vecY,
                 valueType, CUSPARSE_SPMV_ALG_DEFAULT, m_cudaData.d_buffer);
}

template<class TMatrix, class TVector>
typename CudaCGLinearSolver<TMatrix, TVector>::Real
CudaCGLinearSolver<TMatrix, TVector>::gpuDot(void* d_a, void* d_b, int n)
{
    Real result = 0.0;
    cublasHandle_t handle = getCublasCtx();

    if constexpr (sizeof(Real) == sizeof(float))
    {
        cublasSdot(handle, n, static_cast<const float*>(d_a), 1,
                   static_cast<const float*>(d_b), 1, reinterpret_cast<float*>(&result));
    }
    else
    {
        cublasDdot(handle, n, static_cast<const double*>(d_a), 1,
                   static_cast<const double*>(d_b), 1, &result);
    }

    return result;
}

template<class TMatrix, class TVector>
void CudaCGLinearSolver<TMatrix, TVector>::gpuAxpy(Real alpha, void* d_x, void* d_y, int n)
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
void CudaCGLinearSolver<TMatrix, TVector>::gpuCopy(void* d_src, void* d_dst, int n)
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
void CudaCGLinearSolver<TMatrix, TVector>::gpuScale(Real alpha, void* d_x, int n)
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
void CudaCGLinearSolver<TMatrix, TVector>::solve(Matrix& A, Vector& x, Vector& b)
{
    sofa::helper::AdvancedTimer::stepBegin("CudaCG-Solve");

    const int n = static_cast<int>(b.size());

    msg_info() << "b = " << b;

    // Upload matrix to GPU
    uploadMatrix(A);

    // Initialize graph for residuals
    std::map<std::string, sofa::type::vector<Real>>& graph = *d_graph.beginEdit();
    sofa::type::vector<Real>& graph_error = graph[std::string("Error")];
    graph_error.clear();
    graph_error.push_back(1);
    sofa::type::vector<Real>& graph_den = graph[std::string("Denominator")];
    graph_den.clear();

    // Upload vectors to GPU
    uploadVector(b, m_cudaData.d_b);

    Real rho, rho_1 = 0, alpha, beta;
    unsigned nb_iter = 0;
    const char* endcond = "iterations";

    // Compute initial residual r depending on warmStart option
    if (d_warmStart.getValue())
    {
        uploadVector(x, m_cudaData.d_x);
        // r = b - A*x
        sparseMatVec(m_cudaData.d_x, m_cudaData.d_r);  // r = A*x
        gpuScale(Real(-1.0), m_cudaData.d_r, n);       // r = -A*x
        gpuAxpy(Real(1.0), m_cudaData.d_b, m_cudaData.d_r, n);  // r = b - A*x
    }
    else
    {
        // x = 0, r = b
        mycudaMemset(m_cudaData.d_x, 0, n * sizeof(Real));
        gpuCopy(m_cudaData.d_b, m_cudaData.d_r, n);
    }

    // Compute norm of b
    const Real normb = std::sqrt(gpuDot(m_cudaData.d_b, m_cudaData.d_b, n));

    // Check if forces in LHS vector are non-zero
    if (normb != 0.0)
    {
        for (nb_iter = 1; nb_iter <= d_maxIter.getValue(); ++nb_iter)
        {
            // Compute rho = r . r
            rho = gpuDot(m_cudaData.d_r, m_cudaData.d_r, n);

            // Compute error
            const Real normr = std::sqrt(rho);
            const Real err = normr / normb;
            graph_error.push_back(err);

            // Check tolerance criterion
            if (err <= d_tolerance.getValue())
            {
                if (nb_iter == 1 && m_timeStepCount == 0)
                {
                    msg_warning() << "tolerance reached at first iteration of CG, "
                                  << "check the 'tolerance' data field";
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

            // Compute p
            if (nb_iter == 1)
            {
                // p = r
                gpuCopy(m_cudaData.d_r, m_cudaData.d_p, n);
            }
            else
            {
                // beta = rho / rho_1
                beta = rho / rho_1;
                // p = r + beta * p
                gpuScale(beta, m_cudaData.d_p, n);
                gpuAxpy(Real(1.0), m_cudaData.d_r, m_cudaData.d_p, n);
            }

            // Compute Ap = A * p
            sparseMatVec(m_cudaData.d_p, m_cudaData.d_Ap);

            // Compute denominator: den = p . Ap
            const Real den = gpuDot(m_cudaData.d_p, m_cudaData.d_Ap, n);
            graph_den.push_back(den);

            if (den != 0.0)
            {
                // Check threshold criterion
                if (std::fabs(den) <= d_smallDenominatorThreshold.getValue())
                {
                    if (nb_iter == 1 && m_timeStepCount == 0)
                    {
                        msg_warning() << "denominator threshold reached at first iteration of CG, "
                                      << "check the 'threshold' data field";
                    }
                    else
                    {
                        if (nb_iter == 1 && !m_equilibriumReached)
                        {
                            msg_info() << "Equilibrium reached regarding threshold";
                            m_equilibriumReached = true;
                        }
                        if (nb_iter > 1)
                        {
                            m_equilibriumReached = false;
                        }
                        endcond = "threshold";
                        msg_info() << "den = " << den << ", threshold = " << d_smallDenominatorThreshold.getValue();
                        break;
                    }
                }

                // Compute alpha = rho / den
                alpha = rho / den;

                // Update x: x = x + alpha * p
                gpuAxpy(alpha, m_cudaData.d_p, m_cudaData.d_x, n);

                // Update r: r = r - alpha * Ap
                gpuAxpy(-alpha, m_cudaData.d_Ap, m_cudaData.d_r, n);
            }
            else
            {
                msg_warning() << "den = 0.0, breaking iterations";
                break;
            }

            rho_1 = rho;
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

    sofa::helper::AdvancedTimer::valSet("CudaCG iterations", nb_iter);
    sofa::helper::AdvancedTimer::stepEnd("CudaCG-Solve");

    msg_info() << "solve, nbiter = " << nb_iter << " stop because of " << endcond;
    msg_info() << "solve, solution = " << x;
}

// Explicit template instantiations
template class SOFACUDA_COMPONENT_API CudaCGLinearSolver<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;

template class SOFACUDA_COMPONENT_API CudaCGLinearSolver<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;

// Component registration
void registerCudaCGLinearSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        sofa::core::ObjectRegistrationData(
            "CUDA-accelerated Conjugate Gradient linear solver for assembled sparse matrices. "
            "Uses cuSPARSE for SpMV and cuBLAS for vector operations. "
            "For Vec3 FEM, use CompressedRowSparseMatrixMat3x3d for optimized BSR format.")
        .add<CudaCGLinearSolver<linearalgebra::CompressedRowSparseMatrix<SReal>,
                                linearalgebra::FullVector<SReal>>>()
        .add<CudaCGLinearSolver<linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
                                linearalgebra::FullVector<SReal>>>()
    );
}

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
