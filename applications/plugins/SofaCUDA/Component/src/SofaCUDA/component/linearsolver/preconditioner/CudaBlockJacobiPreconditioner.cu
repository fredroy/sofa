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

#if defined(__cplusplus)
namespace sofa::gpu::cuda
{
#endif

/**
 * Block Jacobi preconditioner kernel: z = D_block^{-1} * r
 * Each thread handles one 3x3 block (one node in Vec3 FEM).
 * invBlocks: nBlocks * 9 values (row-major 3x3 matrices)
 * r: input vector (nBlocks * 3)
 * z: output vector (nBlocks * 3)
 */
template<typename Real>
__global__ void CudaBlockJacobiPreconditioner_apply3x3_kernel(
    int nBlocks,
    const Real* __restrict__ invBlocks,
    const Real* __restrict__ r,
    Real* __restrict__ z)
{
    const int blockIdx_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx_x >= nBlocks) return;

    // Load the inverted 3x3 block (row-major)
    const Real* inv = invBlocks + blockIdx_x * 9;

    // Load input vector for this block
    const Real* r_block = r + blockIdx_x * 3;
    Real r0 = r_block[0];
    Real r1 = r_block[1];
    Real r2 = r_block[2];

    // Matrix-vector multiply: z = inv * r
    Real* z_block = z + blockIdx_x * 3;
    z_block[0] = inv[0] * r0 + inv[1] * r1 + inv[2] * r2;
    z_block[1] = inv[3] * r0 + inv[4] * r1 + inv[5] * r2;
    z_block[2] = inv[6] * r0 + inv[7] * r1 + inv[8] * r2;
}

extern "C"
{

void CudaBlockJacobiPreconditioner_apply3x3f(int nBlocks, const float* invBlocks, const float* r, float* z)
{
    const int threadsPerBlock = 256;
    const int numBlocks = (nBlocks + threadsPerBlock - 1) / threadsPerBlock;

    CudaBlockJacobiPreconditioner_apply3x3_kernel<float><<<numBlocks, threadsPerBlock>>>(
        nBlocks, invBlocks, r, z);
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void CudaBlockJacobiPreconditioner_apply3x3d(int nBlocks, const double* invBlocks, const double* r, double* z)
{
    const int threadsPerBlock = 256;
    const int numBlocks = (nBlocks + threadsPerBlock - 1) / threadsPerBlock;

    CudaBlockJacobiPreconditioner_apply3x3_kernel<double><<<numBlocks, threadsPerBlock>>>(
        nBlocks, invBlocks, r, z);
}
#endif

} // extern "C"

#if defined(__cplusplus)
} // namespace sofa::gpu::cuda
#endif
