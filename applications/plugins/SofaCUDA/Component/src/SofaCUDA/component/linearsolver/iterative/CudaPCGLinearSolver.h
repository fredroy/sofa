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
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/helper/map.h>
#include <SofaCUDA/component/linearsolver/preconditioner/CudaPreconditioner.h>

#include <cusparse_v2.h>
#include <cublas_v2.h>

namespace sofa::gpu::cuda
{

/**
 * @brief CUDA-accelerated Preconditioned Conjugate Gradient linear solver.
 *
 * This solver uses cuSPARSE for sparse matrix-vector multiplication and cuBLAS
 * for vector operations. It supports linking to a preconditioner (such as
 * CudaJacobiPreconditioner or CudaIC0Preconditioner) via the 'preconditioner' link.
 *
 * The preconditioner must also be a CUDA-based solver to avoid GPU-CPU transfers
 * during the solve loop.
 */
template<class TMatrix, class TVector>
class CudaPCGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CudaPCGLinearSolver, TMatrix, TVector),
               SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, TMatrix, TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector> Inherit;
    typedef typename TVector::Real Real;

    Data<unsigned> d_maxIter; ///< Maximum number of iterations
    Data<Real> d_tolerance; ///< Desired accuracy (ratio of current residual norm over initial residual norm)
    Data<Real> d_smallDenominatorThreshold; ///< Minimum value of the denominator in CG
    Data<bool> d_warmStart; ///< Use previous solution as initial solution
    Data<bool> d_usePreconditioner; ///< Use preconditioner
    Data<std::map<std::string, sofa::type::vector<Real>>> d_graph; ///< Graph of residuals at each iteration

    SingleLink<CudaPCGLinearSolver, sofa::core::objectmodel::BaseObject,
               BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_preconditioner;

protected:
    CudaPCGLinearSolver();
    ~CudaPCGLinearSolver() override;

public:
    void init() override;
    void reinit() override {}

    /// Solve the linear system Ax=b using preconditioned conjugate gradient on GPU
    void solve(Matrix& A, Vector& x, Vector& b) override;

private:
    /// GPU resources for cuSPARSE
    struct CudaSparseData
    {
        // CSR matrix data on GPU
        void* d_csrVal = nullptr;
        void* d_csrRowPtr = nullptr;
        void* d_csrColInd = nullptr;

        // Vectors on GPU
        void* d_x = nullptr;
        void* d_b = nullptr;
        void* d_r = nullptr;
        void* d_z = nullptr;  // preconditioned residual
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
        int nRows = 0;
        int nCols = 0;
        int nnz = 0;

        bool allocated = false;
    };

    CudaSparseData m_cudaData;

    /// Allocate GPU resources
    void allocateCudaResources(int nRows, int nCols, int nnz);

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
    int m_newtonIter = 0;
};

#if !defined(SOFACUDA_CUDAPCGLINEARSOLVER_CPP)
extern template class SOFACUDA_COMPONENT_API CudaPCGLinearSolver<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;
extern template class SOFACUDA_COMPONENT_API CudaPCGLinearSolver<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;
#endif

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
