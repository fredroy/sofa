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
#include <sofa/linearalgebra/BaseMatrix.h>

namespace sofa::gpu::cuda
{

/**
 * @brief Interface for CUDA preconditioners that can operate directly on GPU buffers.
 *
 * This interface allows CudaPCGLinearSolver to call preconditioners without
 * CPU-GPU data transfers during the solve loop.
 */
class SOFACUDA_COMPONENT_API CudaPreconditionerBase
{
public:
    virtual ~CudaPreconditionerBase() = default;

    /// Build/update the preconditioner from the given matrix
    /// This is called once before the CG iterations begin
    virtual void updatePreconditioner(sofa::linearalgebra::BaseMatrix* matrix) = 0;

    /// Apply the preconditioner directly on GPU buffers: d_z = M^{-1} * d_r
    /// @param d_z Output vector on GPU (preconditioned result)
    /// @param d_r Input vector on GPU (residual)
    /// @param n Size of the vectors
    virtual void solveOnGPU(void* d_z, const void* d_r, int n) = 0;

    /// Check if the preconditioner is ready for GPU solve
    virtual bool isReadyForGPU() const = 0;
};

} // namespace sofa::gpu::cuda
