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

#include <sofa/core/objectmodel/BaseComponent.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullVector.h>
#include <SofaCUDA/component/linearsolver/preconditioner/CudaPreconditioner.h>

namespace sofa::gpu::cuda
{

/**
 * @brief CUDA-accelerated Block Jacobi preconditioner.
 *
 * This preconditioner extracts and inverts the diagonal blocks of the matrix.
 * For 3D FEM problems (Vec3), it inverts 3x3 blocks, capturing the x-y-z coupling
 * at each node. This is more effective than scalar Jacobi for coupled systems.
 *
 * z = D_block^{-1} * r
 *
 * where D_block contains the inverted diagonal blocks.
 *
 * Usage: Link this preconditioner to a CudaPCGLinearSolver via the
 * 'preconditioner' link.
 */
template<class TMatrix, class TVector>
class CudaBlockJacobiPreconditioner : public sofa::core::objectmodel::BaseObject,
                                       public CudaPreconditionerBase
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CudaBlockJacobiPreconditioner, TMatrix, TVector),
               sofa::core::objectmodel::BaseObject);

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::core::objectmodel::BaseObject Inherit;
    typedef typename TVector::Real Real;

protected:
    CudaBlockJacobiPreconditioner();
    ~CudaBlockJacobiPreconditioner() override;

public:
    void init() override;

    /// Build the preconditioner from the matrix (extract and invert diagonal blocks)
    void invert(Matrix& M);

    /// Apply the preconditioner directly on GPU buffers: d_z = D_block^{-1} * d_r
    void solveOnGPU(void* d_z, const void* d_r, int n) override;

    /// Check if the preconditioner is ready for GPU solve
    bool isReadyForGPU() const override { return m_d_invBlocks != nullptr && m_nBlocks > 0; }

    /// Build/update the preconditioner from the given matrix
    void updatePreconditioner(sofa::linearalgebra::BaseMatrix* matrix) override;

private:
    void* m_d_invBlocks = nullptr;  ///< Inverted 3x3 blocks on GPU (stored as 9 * nBlocks floats/doubles)
    int m_nBlocks = 0;              ///< Number of blocks
    int m_blockSize = 3;            ///< Block size (3 for Vec3)

    void freeCudaResources();
};

#if !defined(SOFACUDA_CUDABLOCKJACOBIPRECONDITIONER_CPP)
extern template class SOFACUDA_COMPONENT_API CudaBlockJacobiPreconditioner<
    linearalgebra::CompressedRowSparseMatrix<SReal>,
    linearalgebra::FullVector<SReal>>;
extern template class SOFACUDA_COMPONENT_API CudaBlockJacobiPreconditioner<
    linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>,
    linearalgebra::FullVector<SReal>>;
#endif

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUBLAS
