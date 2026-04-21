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
#include <sofa/gpu/cuda/CudaBaseVector.h>
#include <sofa/helper/map.h>

#include <cusparse_v2.h>
#include <cublas_v2.h>

namespace sofa::gpu::cuda
{

/**
 * @brief CUDA-accelerated Conjugate Gradient linear solver for assembled sparse matrices.
 *
 * This solver uses cuSPARSE for sparse matrix-vector multiplication and cuBLAS
 * for vector operations (dot product, axpy). It works with CompressedRowSparseMatrix
 * and transfers data to the GPU for efficient parallel computation.
 *
 * Requirements:
 * - SOFA must be compiled with SOFACUDA_CUBLAS=ON
 * - A CUDA-capable GPU with compute capability >= 3.5
 */
template<class TMatrix, class TVector>
class CudaCGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CudaCGLinearSolver, TMatrix, TVector),
               SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, TMatrix, TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector> Inherit;
    typedef typename TVector::Real Real;

    Data<unsigned> d_maxIter; ///< Maximum number of iterations
    Data<Real> d_tolerance; ///< Desired accuracy (ratio of current residual norm over initial residual norm)
    Data<Real> d_smallDenominatorThreshold; ///< Minimum value of the denominator in CG
    Data<bool> d_warmStart; ///< Use previous solution as initial solution
    Data<std::map<std::string, sofa::type::vector<Real>>> d_graph; ///< Graph of residuals at each iteration

protected:
    CudaCGLinearSolver();
    ~CudaCGLinearSolver() override;

public:
    void init() override;
    void reinit() override {}

    /// Solve the linear system Ax=b using conjugate gradient on GPU
    void solve(Matrix& A, Vector& x, Vector& b) override;

private:
    /// GPU resources for cuSPARSE
    struct CudaSparseData
    {
        // BSR/CSR matrix data on GPU
        void* d_bsrVal = nullptr;     // Block values (for BSR) or scalar values (for CSR)
        void* d_bsrRowPtr = nullptr;  // Row pointers
        void* d_bsrColInd = nullptr;  // Column indices

        // Vectors on GPU
        void* d_x = nullptr;
        void* d_b = nullptr;
        void* d_r = nullptr;
        void* d_p = nullptr;
        void* d_Ap = nullptr;

        // cuSPARSE matrix descriptor (new API)
        cusparseSpMatDescr_t matA = nullptr;

        // cuSPARSE dense vector descriptors
        cusparseDnVecDescr_t vecX = nullptr;
        cusparseDnVecDescr_t vecY = nullptr;

        // Buffer for SpMV
        void* d_buffer = nullptr;
        size_t bufferSize = 0;

        // Matrix dimensions
        int nRows = 0;        // Scalar rows
        int nCols = 0;        // Scalar cols
        int nnz = 0;          // Number of non-zero scalars (CSR) or blocks (BSR)
        int blockDim = 1;     // Block dimension (1 for CSR, 3 for BSR with Vec3)
        int nBlockRows = 0;   // Number of block rows (BSR only)
        int nBlockCols = 0;   // Number of block cols (BSR only)
        bool useBSR = false;  // True if using BSR format

        bool allocated = false;
    };

    CudaSparseData m_cudaData;

    /// Allocate GPU resources for CSR format
    void allocateCudaResourcesCSR(int nRows, int nCols, int nnz);

    /// Allocate GPU resources for BSR format
    void allocateCudaResourcesBSR(int nBlockRows, int nBlockCols, int nnzBlocks, int blockDim);

    /// Free GPU resources
    void freeCudaResources();

    /// Upload matrix to GPU
    void uploadMatrix(const Matrix& A);

    /// Upload vector to GPU
    void uploadVector(const Vector& src, void* d_dst);

    /// Download vector from GPU
    void downloadVector(void* d_src, Vector& dst);

    /// Perform sparse matrix-vector multiplication: y = A*x
    void sparseMatVec(void* d_x, void* d_y);

    /// Compute dot product on GPU
    Real gpuDot(void* d_a, void* d_b, int n);

    /// Perform axpy on GPU: y = alpha*x + y
    void gpuAxpy(Real alpha, void* d_x, void* d_y, int n);

    /// Copy vector on GPU
    void gpuCopy(void* d_src, void* d_dst, int n);

    /// Scale vector on GPU: x = alpha*x
    void gpuScale(Real alpha, void* d_x, int n);

    int m_timeStepCount = 0;
    bool m_equilibriumReached = false;
};

#if !defined(SOFACUDA_CUDACGLINEARSOLVER_CPP)
extern template class SOFACUDA_COMPONENT_API CudaCGLinearSolver<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;
extern template class SOFACUDA_COMPONENT_API CudaCGLinearSolver<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;
#endif

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
