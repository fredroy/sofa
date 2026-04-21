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
#define SOFACUDA_CUDAIC0PRECONDITIONER_CPP

#include <SofaCUDA/component/config.h>

#ifdef SOFA_GPU_CUBLAS

#include <SofaCUDA/component/linearsolver/preconditioner/CudaIC0Preconditioner.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/gpu/cuda/mycuda.h>

#include <algorithm>
#include <vector>
#include <cmath>

namespace sofa::gpu::cuda
{

template<class TMatrix, class TVector>
CudaIC0Preconditioner<TMatrix, TVector>::CudaIC0Preconditioner()
{
}

template<class TMatrix, class TVector>
CudaIC0Preconditioner<TMatrix, TVector>::~CudaIC0Preconditioner()
{
    freeCudaResources();
}

template<class TMatrix, class TVector>
void CudaIC0Preconditioner<TMatrix, TVector>::init()
{
    Inherit::init();
    mycudaInit();
}

template<class TMatrix, class TVector>
void CudaIC0Preconditioner<TMatrix, TVector>::allocateCudaResources(int nRows, int nnz)
{
    if (m_cudaData.allocated &&
        m_cudaData.nRows == nRows &&
        m_cudaData.nnz == nnz)
    {
        return;
    }

    freeCudaResources();

    m_cudaData.nRows = nRows;
    m_cudaData.nnz = nnz;

    // Allocate CSR matrix data
    mycudaMalloc(&m_cudaData.d_csrVal, nnz * sizeof(Real));
    mycudaMalloc(&m_cudaData.d_csrRowPtr, (nRows + 1) * sizeof(int));
    mycudaMalloc(&m_cudaData.d_csrColInd, nnz * sizeof(int));

    // Allocate temporary vector
    mycudaMalloc(&m_cudaData.d_tmp, nRows * sizeof(Real));

    m_cudaData.allocated = true;
    m_cudaData.factorized = false;
    m_cudaData.analysisComplete = false;
}

template<class TMatrix, class TVector>
void CudaIC0Preconditioner<TMatrix, TVector>::freeCudaResources()
{
    if (!m_cudaData.allocated) return;

    // Destroy SpSV descriptors
    if (m_cudaData.spsvDescrL) cusparseSpSV_destroyDescr(m_cudaData.spsvDescrL);
    if (m_cudaData.spsvDescrLt) cusparseSpSV_destroyDescr(m_cudaData.spsvDescrLt);

    // Destroy matrix/vector descriptors
    if (m_cudaData.matL) cusparseDestroySpMat(m_cudaData.matL);
    if (m_cudaData.vecR) cusparseDestroyDnVec(m_cudaData.vecR);
    if (m_cudaData.vecTmp) cusparseDestroyDnVec(m_cudaData.vecTmp);
    if (m_cudaData.vecZ) cusparseDestroyDnVec(m_cudaData.vecZ);

    // Destroy IC0 resources
    if (m_cudaData.descrA) cusparseDestroyMatDescr(m_cudaData.descrA);
    if (m_cudaData.ic02Info) cusparseDestroyCsric02Info(m_cudaData.ic02Info);
    if (m_cudaData.d_ic02Buffer) mycudaFree(m_cudaData.d_ic02Buffer);

    // Free GPU memory
    if (m_cudaData.d_csrVal) mycudaFree(m_cudaData.d_csrVal);
    if (m_cudaData.d_csrRowPtr) mycudaFree(m_cudaData.d_csrRowPtr);
    if (m_cudaData.d_csrColInd) mycudaFree(m_cudaData.d_csrColInd);
    if (m_cudaData.d_tmp) mycudaFree(m_cudaData.d_tmp);
    if (m_cudaData.d_bufferL) mycudaFree(m_cudaData.d_bufferL);
    if (m_cudaData.d_bufferLt) mycudaFree(m_cudaData.d_bufferLt);

    m_cudaData = CudaIC0Data();
}

template<class TMatrix, class TVector>
void CudaIC0Preconditioner<TMatrix, TVector>::uploadMatrixAndFactorize(const Matrix& M)
{
    using Block = typename Matrix::Block;
    constexpr int BlockRows = Matrix::NL;
    constexpr int BlockCols = Matrix::NC;

    const auto& rowBegin = M.getRowBegin();
    const auto& colsIndex = M.getColsIndex();
    const auto& colsValue = M.getColsValue();

    const int blockRows = static_cast<int>(M.nBlockRow);
    const int scalarRows = blockRows * BlockRows;

    // Build scalar CSR from block CSR - only lower triangular part for IC0
    std::vector<Real> values;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;

    rowPtr.reserve(scalarRows + 1);
    rowPtr.push_back(0);

    for (int br = 0; br < blockRows; ++br)
    {
        for (int lr = 0; lr < BlockRows; ++lr)
        {
            const int scalarRow = br * BlockRows + lr;

            // Collect entries for this scalar row (only lower triangular: col <= row)
            std::vector<std::pair<int, Real>> rowEntries;

            for (auto idx = rowBegin[br]; idx < rowBegin[br + 1]; ++idx)
            {
                const int bc = static_cast<int>(colsIndex[idx]);
                const Block& block = colsValue[idx];

                for (int lc = 0; lc < BlockCols; ++lc)
                {
                    const int scalarCol = bc * BlockCols + lc;

                    // Only include lower triangular entries (col <= row)
                    if (scalarCol > scalarRow) continue;

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
    const int n = scalarRows;

    // Allocate GPU resources
    allocateCudaResources(scalarRows, nnz);

    // Upload to GPU
    mycudaMemcpyHostToDevice(m_cudaData.d_csrVal, values.data(), nnz * sizeof(Real));
    mycudaMemcpyHostToDevice(m_cudaData.d_csrRowPtr, rowPtr.data(), (scalarRows + 1) * sizeof(int));
    mycudaMemcpyHostToDevice(m_cudaData.d_csrColInd, colIdx.data(), nnz * sizeof(int));

    cusparseHandle_t handle = getCusparseCtx();
    cusparseStatus_t status;

    // ========== Perform IC0 factorization using legacy API ==========

    // Create/recreate IC0 resources
    if (m_cudaData.descrA) { cusparseDestroyMatDescr(m_cudaData.descrA); m_cudaData.descrA = nullptr; }
    if (m_cudaData.ic02Info) { cusparseDestroyCsric02Info(m_cudaData.ic02Info); m_cudaData.ic02Info = nullptr; }
    if (m_cudaData.d_ic02Buffer) { mycudaFree(m_cudaData.d_ic02Buffer); m_cudaData.d_ic02Buffer = nullptr; }

    cusparseCreateMatDescr(&m_cudaData.descrA);
    cusparseSetMatType(m_cudaData.descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(m_cudaData.descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(m_cudaData.descrA, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(m_cudaData.descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);

    cusparseCreateCsric02Info(&m_cudaData.ic02Info);

    // Get buffer size for IC0
    int bufferSizeInt = 0;
    if constexpr (sizeof(Real) == sizeof(float))
    {
        cusparseScsric02_bufferSize(handle, n, nnz, m_cudaData.descrA,
            static_cast<float*>(m_cudaData.d_csrVal),
            static_cast<int*>(m_cudaData.d_csrRowPtr),
            static_cast<int*>(m_cudaData.d_csrColInd),
            m_cudaData.ic02Info, &bufferSizeInt);
    }
    else
    {
        cusparseDcsric02_bufferSize(handle, n, nnz, m_cudaData.descrA,
            static_cast<double*>(m_cudaData.d_csrVal),
            static_cast<int*>(m_cudaData.d_csrRowPtr),
            static_cast<int*>(m_cudaData.d_csrColInd),
            m_cudaData.ic02Info, &bufferSizeInt);
    }

    m_cudaData.ic02BufferSize = static_cast<size_t>(bufferSizeInt);
    if (m_cudaData.ic02BufferSize > 0)
        mycudaMalloc(&m_cudaData.d_ic02Buffer, m_cudaData.ic02BufferSize);

    // Analyze sparsity pattern for IC0
    if constexpr (sizeof(Real) == sizeof(float))
    {
        status = cusparseScsric02_analysis(handle, n, nnz, m_cudaData.descrA,
            static_cast<float*>(m_cudaData.d_csrVal),
            static_cast<int*>(m_cudaData.d_csrRowPtr),
            static_cast<int*>(m_cudaData.d_csrColInd),
            m_cudaData.ic02Info,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            m_cudaData.d_ic02Buffer);
    }
    else
    {
        status = cusparseDcsric02_analysis(handle, n, nnz, m_cudaData.descrA,
            static_cast<double*>(m_cudaData.d_csrVal),
            static_cast<int*>(m_cudaData.d_csrRowPtr),
            static_cast<int*>(m_cudaData.d_csrColInd),
            m_cudaData.ic02Info,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            m_cudaData.d_ic02Buffer);
    }

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        msg_error() << "csric02_analysis failed with status: " << status;
        return;
    }

    // Check for zero pivot (matrix not SPD or has zero on diagonal)
    int structural_zero = -1;
    cusparseXcsric02_zeroPivot(handle, m_cudaData.ic02Info, &structural_zero);
    if (structural_zero >= 0)
    {
        msg_warning() << "IC0: structural zero at row " << structural_zero;
    }

    // Perform the actual IC0 factorization (modifies d_csrVal in-place)
    if constexpr (sizeof(Real) == sizeof(float))
    {
        status = cusparseScsric02(handle, n, nnz, m_cudaData.descrA,
            static_cast<float*>(m_cudaData.d_csrVal),
            static_cast<int*>(m_cudaData.d_csrRowPtr),
            static_cast<int*>(m_cudaData.d_csrColInd),
            m_cudaData.ic02Info,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            m_cudaData.d_ic02Buffer);
    }
    else
    {
        status = cusparseDcsric02(handle, n, nnz, m_cudaData.descrA,
            static_cast<double*>(m_cudaData.d_csrVal),
            static_cast<int*>(m_cudaData.d_csrRowPtr),
            static_cast<int*>(m_cudaData.d_csrColInd),
            m_cudaData.ic02Info,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            m_cudaData.d_ic02Buffer);
    }

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        msg_error() << "csric02 factorization failed with status: " << status;
        return;
    }

    // Check for numerical zero pivot
    int numerical_zero = -1;
    cusparseXcsric02_zeroPivot(handle, m_cudaData.ic02Info, &numerical_zero);
    if (numerical_zero >= 0)
    {
        msg_warning() << "IC0: numerical zero at row " << numerical_zero << " (matrix may not be SPD)";
    }

    // ========== Set up SpSV for triangular solves using factored L ==========

    cudaDataType valueType = (sizeof(Real) == sizeof(float)) ? CUDA_R_32F : CUDA_R_64F;

    // Destroy old descriptors if they exist
    if (m_cudaData.spsvDescrL) { cusparseSpSV_destroyDescr(m_cudaData.spsvDescrL); m_cudaData.spsvDescrL = nullptr; }
    if (m_cudaData.spsvDescrLt) { cusparseSpSV_destroyDescr(m_cudaData.spsvDescrLt); m_cudaData.spsvDescrLt = nullptr; }
    if (m_cudaData.matL) { cusparseDestroySpMat(m_cudaData.matL); m_cudaData.matL = nullptr; }
    if (m_cudaData.vecR) { cusparseDestroyDnVec(m_cudaData.vecR); m_cudaData.vecR = nullptr; }
    if (m_cudaData.vecTmp) { cusparseDestroyDnVec(m_cudaData.vecTmp); m_cudaData.vecTmp = nullptr; }
    if (m_cudaData.vecZ) { cusparseDestroyDnVec(m_cudaData.vecZ); m_cudaData.vecZ = nullptr; }
    if (m_cudaData.d_bufferL) { mycudaFree(m_cudaData.d_bufferL); m_cudaData.d_bufferL = nullptr; }
    if (m_cudaData.d_bufferLt) { mycudaFree(m_cudaData.d_bufferLt); m_cudaData.d_bufferLt = nullptr; }

    // Create sparse matrix descriptor for L (now contains factored values)
    cusparseCreateCsr(&m_cudaData.matL, n, n, nnz,
                      m_cudaData.d_csrRowPtr, m_cudaData.d_csrColInd, m_cudaData.d_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, valueType);

    // Set matrix attributes for lower triangular with non-unit diagonal
    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;

    cusparseSpMatSetAttribute(m_cudaData.matL, CUSPARSE_SPMAT_FILL_MODE,
                              &fillMode, sizeof(cusparseFillMode_t));
    cusparseSpMatSetAttribute(m_cudaData.matL, CUSPARSE_SPMAT_DIAG_TYPE,
                              &diagType, sizeof(cusparseDiagType_t));

    // Allocate temporary device memory for vector descriptors
    void* d_dummyR = nullptr;
    void* d_dummyZ = nullptr;
    mycudaMalloc(&d_dummyR, n * sizeof(Real));
    mycudaMalloc(&d_dummyZ, n * sizeof(Real));

    // Create dense vector descriptors (with dummy pointers for now)
    cusparseCreateDnVec(&m_cudaData.vecR, n, d_dummyR, valueType);
    cusparseCreateDnVec(&m_cudaData.vecTmp, n, m_cudaData.d_tmp, valueType);
    cusparseCreateDnVec(&m_cudaData.vecZ, n, d_dummyZ, valueType);

    // Create SpSV descriptors for triangular solves
    cusparseSpSV_createDescr(&m_cudaData.spsvDescrL);
    cusparseSpSV_createDescr(&m_cudaData.spsvDescrLt);

    Real alpha = Real(1.0);

    // Get buffer sizes for SpSV (L * tmp = r)
    cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, m_cudaData.matL, m_cudaData.vecR, m_cudaData.vecTmp,
                            valueType, CUSPARSE_SPSV_ALG_DEFAULT, m_cudaData.spsvDescrL,
                            &m_cudaData.bufferSizeL);

    // Get buffer sizes for SpSV (L^T * z = tmp)
    cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
                            &alpha, m_cudaData.matL, m_cudaData.vecTmp, m_cudaData.vecZ,
                            valueType, CUSPARSE_SPSV_ALG_DEFAULT, m_cudaData.spsvDescrLt,
                            &m_cudaData.bufferSizeLt);

    // Allocate buffers
    if (m_cudaData.bufferSizeL > 0)
        mycudaMalloc(&m_cudaData.d_bufferL, m_cudaData.bufferSizeL);
    if (m_cudaData.bufferSizeLt > 0)
        mycudaMalloc(&m_cudaData.d_bufferLt, m_cudaData.bufferSizeLt);

    // Perform analysis for SpSV (this analyzes the sparsity pattern)
    status = cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, m_cudaData.matL, m_cudaData.vecR, m_cudaData.vecTmp,
                          valueType, CUSPARSE_SPSV_ALG_DEFAULT, m_cudaData.spsvDescrL,
                          m_cudaData.d_bufferL);

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        msg_error() << "cusparseSpSV_analysis (L) failed with status: " << status;
    }

    status = cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE,
                          &alpha, m_cudaData.matL, m_cudaData.vecTmp, m_cudaData.vecZ,
                          valueType, CUSPARSE_SPSV_ALG_DEFAULT, m_cudaData.spsvDescrLt,
                          m_cudaData.d_bufferLt);

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        msg_error() << "cusparseSpSV_analysis (L^T) failed with status: " << status;
    }

    // Free dummy vectors
    mycudaFree(d_dummyR);
    mycudaFree(d_dummyZ);

    m_cudaData.analysisComplete = true;
    m_cudaData.factorized = true;

    msg_info() << "IC0 preconditioner built: size=" << n << ", nnz=" << nnz;
}

template<class TMatrix, class TVector>
void CudaIC0Preconditioner<TMatrix, TVector>::invert(Matrix& M)
{
    sofa::helper::AdvancedTimer::stepBegin("CudaIC0-Factorize");
    uploadMatrixAndFactorize(M);
    sofa::helper::AdvancedTimer::stepEnd("CudaIC0-Factorize");
}

template<class TMatrix, class TVector>
void CudaIC0Preconditioner<TMatrix, TVector>::solveOnGPU(void* d_z, const void* d_r, int n)
{
    if (!m_cudaData.factorized || !m_cudaData.analysisComplete)
    {
        msg_error() << "IC0 preconditioner not ready";
        return;
    }

    if (n != m_cudaData.nRows)
    {
        msg_error() << "Size mismatch: expected " << m_cudaData.nRows << ", got " << n;
        return;
    }

    sofa::helper::AdvancedTimer::stepBegin("CudaIC0-Solve");

    cusparseHandle_t handle = getCusparseCtx();
    cudaDataType valueType = (sizeof(Real) == sizeof(float)) ? CUDA_R_32F : CUDA_R_64F;
    Real alpha = Real(1.0);

    // Update vector descriptors with current pointers
    cusparseDnVecSetValues(m_cudaData.vecR, const_cast<void*>(d_r));
    cusparseDnVecSetValues(m_cudaData.vecZ, d_z);

    // Solve L * tmp = r
    cusparseStatus_t status;
    status = cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &alpha, m_cudaData.matL, m_cudaData.vecR, m_cudaData.vecTmp,
                       valueType, CUSPARSE_SPSV_ALG_DEFAULT, m_cudaData.spsvDescrL);

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        msg_error() << "cusparseSpSV_solve (L) failed with status: " << status;
    }

    // Solve L^T * z = tmp
    status = cusparseSpSV_solve(handle, CUSPARSE_OPERATION_TRANSPOSE,
                       &alpha, m_cudaData.matL, m_cudaData.vecTmp, m_cudaData.vecZ,
                       valueType, CUSPARSE_SPSV_ALG_DEFAULT, m_cudaData.spsvDescrLt);

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        msg_error() << "cusparseSpSV_solve (L^T) failed with status: " << status;
    }

    sofa::helper::AdvancedTimer::stepEnd("CudaIC0-Solve");
}

template<class TMatrix, class TVector>
void CudaIC0Preconditioner<TMatrix, TVector>::updatePreconditioner(sofa::linearalgebra::BaseMatrix* matrix)
{
    if (auto* typedMatrix = dynamic_cast<Matrix*>(matrix))
    {
        invert(*typedMatrix);
    }
    else
    {
        msg_error() << "updatePreconditioner: matrix type mismatch";
    }
}

// Explicit template instantiations
template class SOFACUDA_COMPONENT_API CudaIC0Preconditioner<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;

template class SOFACUDA_COMPONENT_API CudaIC0Preconditioner<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;

// Component registration
void registerCudaIC0Preconditioner(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        sofa::core::ObjectRegistrationData(
            "CUDA-accelerated Incomplete Cholesky (IC0) preconditioner using cuSPARSE. "
            "Uses the lower triangular part of the matrix for L * L^T factorization. "
            "Suitable for symmetric positive definite matrices.")
        .add<CudaIC0Preconditioner<linearalgebra::CompressedRowSparseMatrix<SReal>,
                                    linearalgebra::FullVector<SReal>>>()
        .add<CudaIC0Preconditioner<linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
                                    linearalgebra::FullVector<SReal>>>()
    );
}

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
