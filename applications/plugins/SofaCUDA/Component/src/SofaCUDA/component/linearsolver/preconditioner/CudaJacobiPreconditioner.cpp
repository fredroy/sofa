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
#define SOFACUDA_CUDAJACOBIPRECONDITIONER_CPP

#include <SofaCUDA/component/config.h>

#ifdef SOFA_GPU_CUBLAS

#include <SofaCUDA/component/linearsolver/preconditioner/CudaJacobiPreconditioner.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/gpu/cuda/mycuda.h>
#include <cuda_runtime.h>

namespace sofa::gpu::cuda
{

extern "C"
{
    void CudaJacobiPreconditioner_applyf(int n, const float* invDiag, const float* r, float* z);
#ifdef SOFA_GPU_CUDA_DOUBLE
    void CudaJacobiPreconditioner_applyd(int n, const double* invDiag, const double* r, double* z);
#endif
}

template<class TMatrix, class TVector>
CudaJacobiPreconditioner<TMatrix, TVector>::CudaJacobiPreconditioner()
{
}

template<class TMatrix, class TVector>
CudaJacobiPreconditioner<TMatrix, TVector>::~CudaJacobiPreconditioner()
{
    freeCudaResources();
}

template<class TMatrix, class TVector>
void CudaJacobiPreconditioner<TMatrix, TVector>::init()
{
    BaseObject::init();
    mycudaInit();
}

template<class TMatrix, class TVector>
void CudaJacobiPreconditioner<TMatrix, TVector>::freeCudaResources()
{
    if (m_d_invDiag)
    {
        mycudaFree(m_d_invDiag);
        m_d_invDiag = nullptr;
    }
    m_size = 0;
}

template<class TMatrix, class TVector>
void CudaJacobiPreconditioner<TMatrix, TVector>::invert(Matrix& M)
{
    using Block = typename Matrix::Block;
    constexpr int BlockRows = Matrix::NL;
    constexpr int BlockCols = Matrix::NC;

    const int blockRows = static_cast<int>(M.nBlockRow);
    const int scalarSize = blockRows * BlockRows;

    // Extract diagonal and compute inverse
    // Initialize to 1.0 (identity preconditioner) for missing diagonals
    std::vector<Real> invDiag(scalarSize, Real(1.0));

    const auto& rowBegin = M.getRowBegin();
    const auto& colsIndex = M.getColsIndex();
    const auto& colsValue = M.getColsValue();

    for (int br = 0; br < blockRows; ++br)
    {
        // Find diagonal block
        bool foundDiag = false;
        for (auto idx = rowBegin[br]; idx < rowBegin[br + 1]; ++idx)
        {
            if (static_cast<int>(colsIndex[idx]) == br)
            {
                // Found diagonal block
                const Block& block = colsValue[idx];
                foundDiag = true;

                for (int lr = 0; lr < BlockRows; ++lr)
                {
                    Real diagVal;
                    if constexpr (BlockRows == 1 && BlockCols == 1)
                    {
                        diagVal = static_cast<Real>(block);
                    }
                    else
                    {
                        diagVal = static_cast<Real>(block(lr, lr));
                    }

                    const int scalarIdx = br * BlockRows + lr;
                    if (std::abs(diagVal) > std::numeric_limits<Real>::epsilon())
                    {
                        invDiag[scalarIdx] = Real(1.0) / diagVal;
                    }
                    else
                    {
                        msg_warning() << "Zero or near-zero diagonal element at index " << scalarIdx;
                        invDiag[scalarIdx] = Real(1.0);
                    }
                }
                break;
            }
        }
        if (!foundDiag)
        {
            msg_warning() << "No diagonal block found for block row " << br << ", using identity";
        }
    }

    // Allocate GPU memory if needed
    if (m_size != scalarSize)
    {
        freeCudaResources();
        m_size = scalarSize;
        mycudaMalloc(&m_d_invDiag, m_size * sizeof(Real));
    }

    // Upload to GPU
    mycudaMemcpyHostToDevice(m_d_invDiag, invDiag.data(), m_size * sizeof(Real));

    msg_info() << "Jacobi preconditioner built: size=" << m_size;
}

template<class TMatrix, class TVector>
void CudaJacobiPreconditioner<TMatrix, TVector>::solveOnGPU(void* d_z, const void* d_r, int n)
{
    if (!m_d_invDiag || m_size == 0)
    {
        msg_error() << "solveOnGPU called but preconditioner not ready";
        return;
    }

    if (n != m_size)
    {
        msg_error() << "solveOnGPU: size mismatch - preconditioner size=" << m_size << ", vector size=" << n;
        return;
    }

    if constexpr (sizeof(Real) == sizeof(float))
    {
        CudaJacobiPreconditioner_applyf(n,
            static_cast<const float*>(m_d_invDiag),
            static_cast<const float*>(d_r),
            static_cast<float*>(d_z));
    }
    else
    {
#ifdef SOFA_GPU_CUDA_DOUBLE
        CudaJacobiPreconditioner_applyd(n,
            static_cast<const double*>(m_d_invDiag),
            static_cast<const double*>(d_r),
            static_cast<double*>(d_z));
#else
        msg_error() << "Double precision CUDA not enabled (SOFA_GPU_CUDA_DOUBLE not defined)";
#endif
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        msg_error() << "CUDA kernel error: " << cudaGetErrorString(err);
    }
}

template<class TMatrix, class TVector>
void CudaJacobiPreconditioner<TMatrix, TVector>::updatePreconditioner(sofa::linearalgebra::BaseMatrix* matrix)
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
template class SOFACUDA_COMPONENT_API CudaJacobiPreconditioner<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;

template class SOFACUDA_COMPONENT_API CudaJacobiPreconditioner<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;

// Component registration
void registerCudaJacobiPreconditioner(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        sofa::core::ObjectRegistrationData(
            "CUDA-accelerated Jacobi (diagonal) preconditioner. "
            "Computes z = D^{-1} * r where D is the diagonal of the matrix.")
        .add<CudaJacobiPreconditioner<linearalgebra::CompressedRowSparseMatrix<SReal>,
                                       linearalgebra::FullVector<SReal>>>()
        .add<CudaJacobiPreconditioner<linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
                                       linearalgebra::FullVector<SReal>>>()
    );
}

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
