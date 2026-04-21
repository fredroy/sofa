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
#pragma once

#include <SofaCUDA/component/config.h>
#include <SofaCUDA/core/config.h>

#ifdef SOFA_GPU_CUBLAS

#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullVector.h>
#include <SofaCUDA/component/linearsolver/preconditioner/CudaPreconditioner.h>

#include <cusparse_v2.h>

namespace sofa::gpu::cuda
{

/**
 * @brief CUDA-accelerated Incomplete Cholesky (IC0) preconditioner.
 *
 * This preconditioner computes an incomplete Cholesky factorization of the
 * matrix using cuSPARSE. It is suitable for symmetric positive definite (SPD)
 * matrices, which is typical for FEM stiffness matrices.
 *
 * The preconditioner solves: L * L^T * z = r
 * where L is the incomplete Cholesky factor.
 *
 * Uses the modern cuSPARSE generic API (cusparseSpSV) compatible with CUDA 12+.
 *
 * Usage: Link this preconditioner to a CudaPCGLinearSolver via the
 * 'preconditioner' link.
 */
template<class TMatrix, class TVector>
class CudaIC0Preconditioner : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>,
                               public CudaPreconditionerBase
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CudaIC0Preconditioner, TMatrix, TVector),
               SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, TMatrix, TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector> Inherit;
    typedef typename TVector::Real Real;

protected:
    CudaIC0Preconditioner();
    ~CudaIC0Preconditioner() override;

public:
    void init() override;

    /// Build the preconditioner (compute IC0 factorization)
    void invert(Matrix& M) override;

    /// Apply the preconditioner: solve L * L^T * z = r
    void solve(Matrix& M, Vector& z, Vector& r) override;

    /// Apply the preconditioner directly on GPU buffers: solve L * L^T * d_z = d_r
    /// This avoids CPU-GPU transfers when called from CudaPCGLinearSolver
    void solveOnGPU(void* d_z, const void* d_r, int n) override;

    /// Check if preconditioner is ready for GPU solve
    bool isReadyForGPU() const override { return m_cudaData.factorized; }

    /// Build/update the preconditioner from the given matrix
    void updatePreconditioner(sofa::linearalgebra::BaseMatrix* matrix) override;

private:
    struct CudaIC0Data
    {
        // CSR matrix data on GPU (stores L factor after factorization)
        void* d_csrVal = nullptr;
        void* d_csrRowPtr = nullptr;
        void* d_csrColInd = nullptr;

        // Temporary vectors
        void* d_r = nullptr;
        void* d_z = nullptr;
        void* d_tmp = nullptr;

        // cuSPARSE descriptors for generic API
        cusparseSpMatDescr_t matL = nullptr;      // Lower triangular L
        cusparseSpMatDescr_t matLt = nullptr;     // L^T (for transpose solve)
        cusparseDnVecDescr_t vecR = nullptr;      // Input vector
        cusparseDnVecDescr_t vecTmp = nullptr;    // Temporary vector
        cusparseDnVecDescr_t vecZ = nullptr;      // Output vector

        // SpSV descriptors for triangular solves
        cusparseSpSVDescr_t spsvDescrL = nullptr;
        cusparseSpSVDescr_t spsvDescrLt = nullptr;

        // Buffer for operations
        void* d_buffer = nullptr;
        void* d_bufferL = nullptr;
        void* d_bufferLt = nullptr;
        size_t bufferSize = 0;
        size_t bufferSizeL = 0;
        size_t bufferSizeLt = 0;

        // Matrix dimensions
        int nRows = 0;
        int nnz = 0;

        bool allocated = false;
        bool factorized = false;
        bool analysisComplete = false;
    };

    CudaIC0Data m_cudaData;

    void allocateCudaResources(int nRows, int nnz);
    void freeCudaResources();
    void uploadMatrix(const Matrix& M);
};

#if !defined(SOFACUDA_CUDAIC0PRECONDITIONER_CPP)
extern template class SOFACUDA_COMPONENT_API CudaIC0Preconditioner<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;
extern template class SOFACUDA_COMPONENT_API CudaIC0Preconditioner<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;
#endif

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
