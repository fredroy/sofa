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
#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>

extern "C"
{
    void CudaJacobiPreconditioner_applyf(int n, const float* invDiag, const float* r, float* z);
#if defined(SOFA_GPU_CUDA_DOUBLE)
    void CudaJacobiPreconditioner_applyd(int n, const double* invDiag, const double* r, double* z);
#endif
}

template<typename Real>
__global__ void JacobiPreconditioner_apply_kernel(int n, const Real* invDiag, const Real* r, Real* z)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        z[idx] = invDiag[idx] * r[idx];
    }
}

void CudaJacobiPreconditioner_applyf(int n, const float* invDiag, const float* r, float* z)
{
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    JacobiPreconditioner_apply_kernel<float><<<numBlocks, blockSize>>>(n, invDiag, r, z);
}

#if defined(SOFA_GPU_CUDA_DOUBLE)
void CudaJacobiPreconditioner_applyd(int n, const double* invDiag, const double* r, double* z)
{
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    JacobiPreconditioner_apply_kernel<double><<<numBlocks, blockSize>>>(n, invDiag, r, z);
}
#endif
