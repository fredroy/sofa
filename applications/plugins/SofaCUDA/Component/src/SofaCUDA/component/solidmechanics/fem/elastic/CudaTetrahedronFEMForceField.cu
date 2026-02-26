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
#include "cuda.h"
#include <sofa/gpu/cuda/mycuda.h>

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, const void* elems, void* state, void* f, const void* x);
    void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, const void* elems, const void* state, void* df, const void* dx, double kFactor);
    void TetrahedronFEMForceFieldCuda3f_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
    void TetrahedronFEMForceFieldCuda3f_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);

#ifdef SOFA_GPU_CUDA_DOUBLE
    void TetrahedronFEMForceFieldCuda3d_addForce(unsigned int nbElem, const void* elems, void* state, void* f, const void* x);
    void TetrahedronFEMForceFieldCuda3d_addDForce(unsigned int nbElem, const void* elems, const void* state, void* df, const void* dx, double kFactor);
    void TetrahedronFEMForceFieldCuda3d_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
    void TetrahedronFEMForceFieldCuda3d_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);
#endif // SOFA_GPU_CUDA_DOUBLE
}

template<class real>
class __align__(16) GPUElement
{
public:
    /// index of the 4 connected vertices
    int ia[BSIZE];
    int ib[BSIZE];
    int ic[BSIZE];
    int id[BSIZE];
    /// material stiffness matrix
    real gamma_bx2[BSIZE], half_mu2_bx2[BSIZE];
    /// initial position of the vertices in the local (rotated) coordinate system
    real bx[BSIZE],cx[BSIZE];
    real cy[BSIZE],dx[BSIZE],dy[BSIZE],dz[BSIZE];
    /// strain-displacement matrix
    real Jbx_bx[BSIZE],Jby_bx[BSIZE],Jbz_bx[BSIZE];
};

//////////////////////
// GPU-side methods //
//////////////////////

constexpr int BLOCK_SIZE = 128;

template<typename real>
__device__ __forceinline__
CudaVec3<real> loadPos(const CudaVec3<real>* __restrict__ pos, int idx)
{
    return CudaVec3<real>::make(
        __ldg(&pos[idx].x),
        __ldg(&pos[idx].y),
        __ldg(&pos[idx].z));
}

template<typename real>
__device__ __forceinline__
void atomicAddVec3(CudaVec3<real>* out, int idx, CudaVec3<real> v)
{
    atomicAdd(&out[idx].x, v.x);
    atomicAdd(&out[idx].y, v.y);
    atomicAdd(&out[idx].z, v.z);
}

template<typename real>
__device__ __forceinline__
void atomicSubVec3(CudaVec3<real>* out, int idx, CudaVec3<real> v)
{
    atomicAdd(&out[idx].x, -v.x);
    atomicAdd(&out[idx].y, -v.y);
    atomicAdd(&out[idx].z, -v.z);
}

template<typename real>
__global__ void __launch_bounds__(BLOCK_SIZE)
calcForce_kernel(
    int nbElem,
    const GPUElement<real>* __restrict__ elems,
    real* __restrict__ rotations,
    const CudaVec3<real>* __restrict__ x,
    CudaVec3<real>* f)
{
    const int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= nbElem) return;

    // Access BSIZE-blocked element data
    const GPUElement<real>* e = elems + (elem / BSIZE);
    const int li = elem % BSIZE;

    // Read vertex indices via __ldg
    const int va = __ldg(&e->ia[li]);
    const int vb = __ldg(&e->ib[li]);
    const int vc = __ldg(&e->ic[li]);
    const int vd = __ldg(&e->id[li]);

    // Load current positions
    CudaVec3<real> A = loadPos(x, va);
    CudaVec3<real> B = loadPos(x, vb);
    B -= A;

    // Compute R
    matrix3<real> Rt;
    real bx = norm2(B);
    real inv_bx = rsqrt(bx);
    Rt.x = B * inv_bx;
    bx *= inv_bx;

    // Compute JtRtX = JbtRtB + JctRtC + JdtRtD
    CudaVec3<real> JtRtX0, JtRtX1;

    bx -= __ldg(&e->bx[li]);
    real e_Jbx_bx = __ldg(&e->Jbx_bx[li]);
    real e_Jby_bx = __ldg(&e->Jby_bx[li]);
    real e_Jbz_bx = __ldg(&e->Jbz_bx[li]);
    JtRtX0.x = e_Jbx_bx * bx;
    JtRtX0.y = 0;
    JtRtX0.z = 0;
    JtRtX1.x = e_Jby_bx * bx;
    JtRtX1.y = 0;
    JtRtX1.z = e_Jbz_bx * bx;

    CudaVec3<real> C = loadPos(x, vc);
    C -= A;
    Rt.z = cross(B, C);
    Rt.y = cross(Rt.z, B);
    Rt.y *= invnorm(Rt.y);
    Rt.z *= invnorm(Rt.z);

    real e_cx = __ldg(&e->cx[li]);
    real e_cy = __ldg(&e->cy[li]);
    real cx = Rt.mulX(C) - e_cx;
    real cy = Rt.mulY(C) - e_cy;

    real e_dy = __ldg(&e->dy[li]);
    real e_dz = __ldg(&e->dz[li]);
    //JtRtX0.x += 0;
    JtRtX0.y += e_dz * cy;
    //JtRtX0.z += 0;
    JtRtX1.x += e_dz * cx;
    JtRtX1.y -= e_dy * cy;
    JtRtX1.z -= e_dy * cx;

    CudaVec3<real> D = loadPos(x, vd);
    D -= A;

    real e_dx = __ldg(&e->dx[li]);
    real dx = Rt.mulX(D) - e_dx;
    real dy = Rt.mulY(D) - e_dy;
    real dz = Rt.mulZ(D) - e_dz;

    //JtRtX0.x += 0;
    //JtRtX0.y += 0;
    JtRtX0.z += e_cy * dz;
    //JtRtX1.x += 0;
    JtRtX1.y += e_cy * dy;
    JtRtX1.z += e_cy * dx;

    // Compute S = K JtRtX
    real e_half_mu2_bx2 = __ldg(&e->half_mu2_bx2[li]);
    CudaVec3<real> S0 = JtRtX0 * (e_half_mu2_bx2 + e_half_mu2_bx2);
    S0 += (JtRtX0.x + JtRtX0.y + JtRtX0.z) * __ldg(&e->gamma_bx2[li]);
    CudaVec3<real> S1 = JtRtX1 * e_half_mu2_bx2;

    // Compute element forces (back-rotated)
    CudaVec3<real> fD = Rt.mulT(CudaVec3<real>::make(
            e_cy * S1.z,
            e_cy * S1.y,
            e_cy * S0.z));

    CudaVec3<real> fC = Rt.mulT(CudaVec3<real>::make(
            e_dz * S1.x - e_dy * S1.z,
            e_dz * S0.y - e_dy * S1.y,
            e_dz * S1.y - e_dy * S0.z));

    CudaVec3<real> fB = Rt.mulT(CudaVec3<real>::make(
            e_Jbx_bx * S0.x                   + e_Jby_bx * S1.x                   + e_Jbz_bx * S1.z,
            e_Jby_bx * S0.y                   + e_Jbx_bx * S1.x + e_Jbz_bx * S1.y,
            e_Jbz_bx * S0.z                   + e_Jby_bx * S1.y + e_Jbx_bx * S1.z));

    // Store rotation in BSIZE-interleaved format (for addDForce / getRotations)
    const int rotBase = (elem / BSIZE) * (9 * BSIZE) + li;
    Rt.writeAoS(rotations + rotBase, BSIZE);

    // Scatter forces via atomicAdd (replaces eforce + gather)
    // Original gather did f[v] -= eforce[v_contrib], so:
    //   f[va] += (fB + fC + fD),  f[vb] -= fB,  f[vc] -= fC,  f[vd] -= fD
    atomicAddVec3(f, va, fB + fC + fD);
    atomicSubVec3(f, vb, fB);
    atomicSubVec3(f, vc, fC);
    atomicSubVec3(f, vd, fD);
}

template<typename real>
__global__ void __launch_bounds__(BLOCK_SIZE)
calcDForce_kernel(
    int nbElem,
    const GPUElement<real>* __restrict__ elems,
    const real* __restrict__ rotations,
    const CudaVec3<real>* __restrict__ dx,
    CudaVec3<real>* df,
    real factor)
{
    const int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= nbElem) return;

    // Access BSIZE-blocked element data
    const GPUElement<real>* e = elems + (elem / BSIZE);
    const int li = elem % BSIZE;

    // Read rotation from state buffer (written by prior calcForce call)
    matrix3<real> Rt;
    const int rotBase = (elem / BSIZE) * (9 * BSIZE) + li;
    Rt.readAoS(rotations + rotBase, BSIZE);

    // Read vertex indices via __ldg
    const int va = __ldg(&e->ia[li]);
    const int vb = __ldg(&e->ib[li]);
    const int vc = __ldg(&e->ic[li]);
    const int vd = __ldg(&e->id[li]);

    // Compute JtRtX = JbtRtB + JctRtC + JdtRtD
    CudaVec3<real> A = loadPos(dx, va);
    CudaVec3<real> JtRtX0, JtRtX1;

    CudaVec3<real> B = loadPos(dx, vb);
    B = Rt * (B - A);

    real e_Jbx_bx = __ldg(&e->Jbx_bx[li]);
    real e_Jby_bx = __ldg(&e->Jby_bx[li]);
    real e_Jbz_bx = __ldg(&e->Jbz_bx[li]);
    JtRtX0.x = e_Jbx_bx * B.x;
    JtRtX0.y =                  e_Jby_bx * B.y;
    JtRtX0.z =                                   e_Jbz_bx * B.z;
    JtRtX1.x = e_Jby_bx * B.x + e_Jbx_bx * B.y;
    JtRtX1.y =                  e_Jbz_bx * B.y + e_Jby_bx * B.z;
    JtRtX1.z = e_Jbz_bx * B.x                  + e_Jbx_bx * B.z;

    CudaVec3<real> C = loadPos(dx, vc);
    C = Rt * (C - A);

    real e_dy = __ldg(&e->dy[li]);
    real e_dz = __ldg(&e->dz[li]);
    //JtRtX0.x += 0;
    JtRtX0.y +=              e_dz * C.y;
    JtRtX0.z +=                         - e_dy * C.z;
    JtRtX1.x += e_dz * C.x;
    JtRtX1.y +=            - e_dy * C.y + e_dz * C.z;
    JtRtX1.z -= e_dy * C.x;

    CudaVec3<real> D = loadPos(dx, vd);
    D = Rt * (D - A);

    real e_cy = __ldg(&e->cy[li]);
    //JtRtX0.x += 0;
    //JtRtX0.y += 0;
    JtRtX0.z +=                           e_cy * D.z;
    //JtRtX1.x += 0;
    JtRtX1.y +=              e_cy * D.y;
    JtRtX1.z += e_cy * D.x;

    // Compute S = K JtRtX
    real e_half_mu2_bx2 = __ldg(&e->half_mu2_bx2[li]);
    CudaVec3<real> S0 = JtRtX0 * (e_half_mu2_bx2 + e_half_mu2_bx2);
    S0 += (JtRtX0.x + JtRtX0.y + JtRtX0.z) * __ldg(&e->gamma_bx2[li]);
    CudaVec3<real> S1 = JtRtX1 * e_half_mu2_bx2;

    S0 *= factor;
    S1 *= factor;

    // Compute element forces (back-rotated)
    CudaVec3<real> fD = Rt.mulT(CudaVec3<real>::make(
            e_cy * S1.z,
            e_cy * S1.y,
            e_cy * S0.z));

    CudaVec3<real> fC = Rt.mulT(CudaVec3<real>::make(
            e_dz * S1.x - e_dy * S1.z,
            e_dz * S0.y - e_dy * S1.y,
            e_dz * S1.y - e_dy * S0.z));

    CudaVec3<real> fB = Rt.mulT(CudaVec3<real>::make(
            e_Jbx_bx * S0.x                   + e_Jby_bx * S1.x                   + e_Jbz_bx * S1.z,
            e_Jby_bx * S0.y                   + e_Jbx_bx * S1.x + e_Jbz_bx * S1.y,
            e_Jbz_bx * S0.z                   + e_Jby_bx * S1.y + e_Jbx_bx * S1.z));

    // Scatter forces via atomicAdd
    atomicAddVec3(df, va, fB + fC + fD);
    atomicSubVec3(df, vb, fB);
    atomicSubVec3(df, vc, fC);
    atomicSubVec3(df, vd, fD);
}

template<typename real>
__global__ void __launch_bounds__(BLOCK_SIZE)
getRotations_kernel(int nbVertex, const real* __restrict__ initState, const real* __restrict__ state, const int* __restrict__ rotationIdx, real* rotations)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nbVertex) return;

    const int elemIdx = __ldg(&rotationIdx[index]);
    const int stateIdx = ((elemIdx / BSIZE) * (9 * BSIZE)) + (elemIdx % BSIZE);

    matrix3<real> initRt, curRt, R;

    initRt.readAoS(initState + stateIdx, BSIZE);
    curRt.readAoS(state + stateIdx, BSIZE);

    // R = transpose(curRt) * initRt
    R = curRt.mulT(initRt);

    rotations += 9 * index;
    rotations[0] = R.x.x;
    rotations[1] = R.x.y;
    rotations[2] = R.x.z;

    rotations[3] = R.y.x;
    rotations[4] = R.y.y;
    rotations[5] = R.y.z;

    rotations[6] = R.z.x;
    rotations[7] = R.z.y;
    rotations[8] = R.z.z;
}

template<typename real>
__global__ void __launch_bounds__(BLOCK_SIZE)
getElementRotations_kernel(unsigned nbElem, const real* __restrict__ rotationsAos, real* rotations)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nbElem) return;

    matrix3<real> R;
    const int aosBase = (index / BSIZE) * (9 * BSIZE) + (index % BSIZE);
    R.readAoS(rotationsAos + aosBase, BSIZE);

    rotations += 9 * index;
    rotations[0] = R.x.x;
    rotations[1] = R.y.x;
    rotations[2] = R.z.x;

    rotations[3] = R.x.y;
    rotations[4] = R.y.y;
    rotations[5] = R.z.y;

    rotations[6] = R.x.z;
    rotations[7] = R.y.z;
    rotations[8] = R.z.z;
}

//////////////////////
// CPU-side methods //
//////////////////////

void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, const void* elems, void* state, void* f, const void* x)
{
    dim3 grid((nbElem + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    calcForce_kernel<float><<<grid, threads>>>(
        nbElem,
        (const GPUElement<float>*)elems,
        (float*)state,
        (const CudaVec3<float>*)x,
        (CudaVec3<float>*)f);
    mycudaDebugError("calcForce_kernel<float>");
}

void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, const void* elems, const void* state, void* df, const void* dx, double kFactor)
{
    dim3 grid((nbElem + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    calcDForce_kernel<float><<<grid, threads>>>(
        nbElem,
        (const GPUElement<float>*)elems,
        (const float*)state,
        (const CudaVec3<float>*)dx,
        (CudaVec3<float>*)df,
        (float)kFactor);
    mycudaDebugError("calcDForce_kernel<float>");
}

void TetrahedronFEMForceFieldCuda3f_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
{
    dim3 grid((nbVertex + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    getRotations_kernel<float><<<grid, threads>>>(nbVertex, (const float*)initState, (const float*)state, (const int*)rotationIdx, (float*)rotations);
    mycudaDebugError("getRotations_kernel<float>");
}

void TetrahedronFEMForceFieldCuda3f_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations)
{
    dim3 grid((nbElem + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    getElementRotations_kernel<float><<<grid, threads>>>(nbElem, (const float*)rotationsAos, (float*)rotations);
    mycudaDebugError("getElementRotations_kernel<float>");
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void TetrahedronFEMForceFieldCuda3d_addForce(unsigned int nbElem, const void* elems, void* state, void* f, const void* x)
{
    dim3 grid((nbElem + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    calcForce_kernel<double><<<grid, threads>>>(
        nbElem,
        (const GPUElement<double>*)elems,
        (double*)state,
        (const CudaVec3<double>*)x,
        (CudaVec3<double>*)f);
    mycudaDebugError("calcForce_kernel<double>");
}

void TetrahedronFEMForceFieldCuda3d_addDForce(unsigned int nbElem, const void* elems, const void* state, void* df, const void* dx, double kFactor)
{
    dim3 grid((nbElem + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    calcDForce_kernel<double><<<grid, threads>>>(
        nbElem,
        (const GPUElement<double>*)elems,
        (const double*)state,
        (const CudaVec3<double>*)dx,
        (CudaVec3<double>*)df,
        kFactor);
    mycudaDebugError("calcDForce_kernel<double>");
}

void TetrahedronFEMForceFieldCuda3d_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
{
    dim3 grid((nbVertex + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    getRotations_kernel<double><<<grid, threads>>>(nbVertex, (const double*)initState, (const double*)state, (const int*)rotationIdx, (double*)rotations);
    mycudaDebugError("getRotations_kernel<double>");
}

void TetrahedronFEMForceFieldCuda3d_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations)
{
    dim3 grid((nbElem + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    getElementRotations_kernel<double><<<grid, threads>>>(nbElem, (const double*)rotationsAos, (double*)rotations);
    mycudaDebugError("getElementRotations_kernel<double>");
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
