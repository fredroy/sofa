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
#define SOFACUDA_CUDABLOCKJACOBIPRECONDITIONER_CPP

#include <SofaCUDA/component/config.h>

#ifdef SOFA_GPU_CUBLAS

#include <SofaCUDA/component/linearsolver/preconditioner/CudaBlockJacobiPreconditioner.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/gpu/cuda/mycuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace sofa::gpu::cuda
{

extern "C"
{
    void CudaBlockJacobiPreconditioner_apply3x3f(int nBlocks, const float* invBlocks, const float* r, float* z);
#ifdef SOFA_GPU_CUDA_DOUBLE
    void CudaBlockJacobiPreconditioner_apply3x3d(int nBlocks, const double* invBlocks, const double* r, double* z);
#endif
}

template<class TMatrix, class TVector>
CudaBlockJacobiPreconditioner<TMatrix, TVector>::CudaBlockJacobiPreconditioner()
{
}

template<class TMatrix, class TVector>
CudaBlockJacobiPreconditioner<TMatrix, TVector>::~CudaBlockJacobiPreconditioner()
{
    freeCudaResources();
}

template<class TMatrix, class TVector>
void CudaBlockJacobiPreconditioner<TMatrix, TVector>::init()
{
    Inherit::init();
    mycudaInit();
}

template<class TMatrix, class TVector>
void CudaBlockJacobiPreconditioner<TMatrix, TVector>::freeCudaResources()
{
    if (m_d_invBlocks)
    {
        mycudaFree(m_d_invBlocks);
        m_d_invBlocks = nullptr;
    }
    m_nBlocks = 0;
}

namespace {

template<typename Real>
void invert3x3(const Real* m, Real* inv)
{
    Real det = m[0] * (m[4] * m[8] - m[5] * m[7])
             - m[1] * (m[3] * m[8] - m[5] * m[6])
             + m[2] * (m[3] * m[7] - m[4] * m[6]);

    if (std::abs(det) < std::numeric_limits<Real>::epsilon())
    {
        // Singular matrix - use identity
        for (int i = 0; i < 9; ++i) inv[i] = (i % 4 == 0) ? Real(1) : Real(0);
        return;
    }

    Real invDet = Real(1) / det;

    inv[0] = (m[4] * m[8] - m[5] * m[7]) * invDet;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * invDet;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * invDet;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * invDet;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * invDet;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * invDet;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * invDet;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * invDet;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * invDet;
}

}

template<class TMatrix, class TVector>
void CudaBlockJacobiPreconditioner<TMatrix, TVector>::invert(Matrix& M)
{
    using Block = typename Matrix::Block;
    constexpr int BlockRows = Matrix::NL;
    constexpr int BlockCols = Matrix::NC;

    const int blockRows = static_cast<int>(M.nBlockRow);

    const auto& rowBegin = M.getRowBegin();
    const auto& colsIndex = M.getColsIndex();
    const auto& colsValue = M.getColsValue();

    // For scalar matrices (1x1 blocks), we treat 3 consecutive scalars as a block
    // For 3x3 block matrices, we use the diagonal blocks directly
    constexpr bool isScalarMatrix = (BlockRows == 1 && BlockCols == 1);

    int nBlocks;
    std::vector<Real> invBlocks;

    if constexpr (isScalarMatrix)
    {
        // Scalar matrix: group every 3 rows into a block
        const int scalarRows = blockRows;
        nBlocks = scalarRows / 3;
        if (scalarRows % 3 != 0)
        {
            msg_warning() << "Matrix size " << scalarRows << " is not divisible by 3, truncating";
        }

        invBlocks.resize(nBlocks * 9, Real(0));

        // Build each 3x3 block from the diagonal elements
        for (int b = 0; b < nBlocks; ++b)
        {
            Real block[9] = {0};

            // Extract the 3x3 diagonal block
            for (int lr = 0; lr < 3; ++lr)
            {
                const int row = b * 3 + lr;
                if (row >= blockRows) break;

                for (auto idx = rowBegin[row]; idx < rowBegin[row + 1]; ++idx)
                {
                    const int col = static_cast<int>(colsIndex[idx]);
                    const int lc = col - b * 3;

                    // Only include entries within this 3x3 block
                    if (lc >= 0 && lc < 3)
                    {
                        block[lr * 3 + lc] = static_cast<Real>(colsValue[idx]);
                    }
                }
            }

            // Invert the 3x3 block
            invert3x3(block, &invBlocks[b * 9]);
        }
    }
    else
    {
        // Block matrix: use diagonal blocks directly (assuming 3x3)
        static_assert(BlockRows == 3 && BlockCols == 3, "Block Jacobi currently only supports 3x3 blocks");

        nBlocks = blockRows;
        invBlocks.resize(nBlocks * 9, Real(0));

        for (int br = 0; br < blockRows; ++br)
        {
            Real block[9] = {0};
            bool foundDiag = false;

            // Find diagonal block
            for (auto idx = rowBegin[br]; idx < rowBegin[br + 1]; ++idx)
            {
                if (static_cast<int>(colsIndex[idx]) == br)
                {
                    const Block& diagBlock = colsValue[idx];
                    foundDiag = true;

                    // Copy block to flat array (row-major)
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            block[i * 3 + j] = static_cast<Real>(diagBlock(i, j));
                    break;
                }
            }

            if (!foundDiag)
            {
                // Use identity if no diagonal block found
                block[0] = block[4] = block[8] = Real(1);
            }

            // Invert the 3x3 block
            invert3x3(block, &invBlocks[br * 9]);
        }
    }

    // Allocate GPU memory if needed
    if (m_nBlocks != nBlocks)
    {
        freeCudaResources();
        m_nBlocks = nBlocks;
        mycudaMalloc(&m_d_invBlocks, m_nBlocks * 9 * sizeof(Real));
    }

    // Upload to GPU
    mycudaMemcpyHostToDevice(m_d_invBlocks, invBlocks.data(), m_nBlocks * 9 * sizeof(Real));

    msg_info() << "Block Jacobi preconditioner built: " << m_nBlocks << " blocks (3x3)";
}

template<class TMatrix, class TVector>
void CudaBlockJacobiPreconditioner<TMatrix, TVector>::solveOnGPU(void* d_z, const void* d_r, int n)
{
    if (!m_d_invBlocks || m_nBlocks == 0)
    {
        msg_error() << "solveOnGPU called but preconditioner not ready";
        return;
    }

    const int expectedSize = m_nBlocks * 3;
    if (n != expectedSize)
    {
        msg_error() << "solveOnGPU: size mismatch - expected " << expectedSize << ", got " << n;
        return;
    }

    if constexpr (sizeof(Real) == sizeof(float))
    {
        CudaBlockJacobiPreconditioner_apply3x3f(m_nBlocks,
            static_cast<const float*>(m_d_invBlocks),
            static_cast<const float*>(d_r),
            static_cast<float*>(d_z));
    }
    else
    {
#ifdef SOFA_GPU_CUDA_DOUBLE
        CudaBlockJacobiPreconditioner_apply3x3d(m_nBlocks,
            static_cast<const double*>(m_d_invBlocks),
            static_cast<const double*>(d_r),
            static_cast<double*>(d_z));
#else
        msg_error() << "Double precision CUDA not enabled";
#endif
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        msg_error() << "CUDA kernel error: " << cudaGetErrorString(err);
    }
}

template<class TMatrix, class TVector>
void CudaBlockJacobiPreconditioner<TMatrix, TVector>::updatePreconditioner(sofa::linearalgebra::BaseMatrix* matrix)
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
template class SOFACUDA_COMPONENT_API CudaBlockJacobiPreconditioner<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;

template class SOFACUDA_COMPONENT_API CudaBlockJacobiPreconditioner<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;

// Component registration
void registerCudaBlockJacobiPreconditioner(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        sofa::core::ObjectRegistrationData(
            "CUDA-accelerated Block Jacobi preconditioner. "
            "Inverts 3x3 diagonal blocks, capturing x-y-z coupling for Vec3 FEM problems. "
            "More effective than scalar Jacobi for coupled systems.")
        .add<CudaBlockJacobiPreconditioner<linearalgebra::CompressedRowSparseMatrix<SReal>,
                                            linearalgebra::FullVector<SReal>>>()
        .add<CudaBlockJacobiPreconditioner<linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
                                            linearalgebra::FullVector<SReal>>>()
    );
}

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
