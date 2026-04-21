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

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullVector.h>
#include <SofaCUDA/component/linearsolver/preconditioner/CudaPreconditioner.h>

#include <cublas_v2.h>

namespace sofa::gpu::cuda
{

/**
 * @brief CUDA-accelerated Jacobi (diagonal) preconditioner.
 *
 * This preconditioner extracts the diagonal of the matrix and performs
 * element-wise division on the GPU: z = D^{-1} * r
 *
 * Usage: Link this preconditioner to a CudaPCGLinearSolver via the
 * 'preconditioner' link.
 *
 * Note: This is NOT a linear solver itself - it only provides preconditioning
 * functionality to be used by CudaPCGLinearSolver.
 */
template<class TMatrix, class TVector>
class CudaJacobiPreconditioner : public sofa::core::objectmodel::BaseObject,
                                  public CudaPreconditionerBase
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CudaJacobiPreconditioner, TMatrix, TVector),
               sofa::core::objectmodel::BaseObject);

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::core::objectmodel::BaseObject Inherit;
    typedef typename TVector::Real Real;

protected:
    CudaJacobiPreconditioner();
    ~CudaJacobiPreconditioner() override;

public:
    void init() override;

    /// Build the preconditioner from the matrix (extract and invert diagonal)
    void invert(Matrix& M);

    /// Get the inverse diagonal stored on GPU (for use by CudaPCGLinearSolver)
    const void* getDeviceInverseDiagonal() const { return m_d_invDiag; }

    /// Get the size of the diagonal
    int getDiagonalSize() const { return m_size; }

    /// Check if preconditioner data is on GPU
    bool isOnGPU() const { return m_d_invDiag != nullptr; }

    /// Apply the preconditioner directly on GPU buffers: d_z = D^{-1} * d_r
    /// This avoids CPU-GPU transfers when called from CudaPCGLinearSolver
    void solveOnGPU(void* d_z, const void* d_r, int n) override;

    /// Check if the preconditioner is ready for GPU solve
    bool isReadyForGPU() const override { return m_d_invDiag != nullptr && m_size > 0; }

    /// Build/update the preconditioner from the given matrix
    void updatePreconditioner(sofa::linearalgebra::BaseMatrix* matrix) override;

private:
    void* m_d_invDiag = nullptr;  ///< Inverse diagonal on GPU
    int m_size = 0;               ///< Size of the diagonal

    void freeCudaResources();
};

#if !defined(SOFACUDA_CUDAJACOBIPRECONDITIONER_CPP)
extern template class SOFACUDA_COMPONENT_API CudaJacobiPreconditioner<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;
extern template class SOFACUDA_COMPONENT_API CudaJacobiPreconditioner<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;
#endif

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
