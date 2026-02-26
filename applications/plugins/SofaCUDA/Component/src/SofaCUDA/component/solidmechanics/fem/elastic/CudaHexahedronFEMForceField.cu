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
#include <cuda.h>
#include <sofa/gpu/cuda/mycuda.h>

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

template<class real>
class __align__(16) GPUElement
{
public:
    /// index of the 8 connected vertices
    int ia[BSIZE];
    int ib[BSIZE];
    int ic[BSIZE];
    int id[BSIZE];
    int ig[BSIZE];
    int ih[BSIZE];
    int ii[BSIZE];
    int ij[BSIZE];

    /// initial position of the vertices in the local (rotated) coordinate system
    real ax[BSIZE],ay[BSIZE],az[BSIZE];
    real bx[BSIZE],by[BSIZE],bz[BSIZE];
    real cx[BSIZE],cy[BSIZE],cz[BSIZE];
    real dx[BSIZE],dy[BSIZE],dz[BSIZE];
    real gx[BSIZE],gy[BSIZE],gz[BSIZE];
    real hx[BSIZE],hy[BSIZE],hz[BSIZE];
    real ix[BSIZE],iy[BSIZE],iz[BSIZE];
    real jx[BSIZE],jy[BSIZE],jz[BSIZE];

};

/// Symmetric element stiffness matrix: upper triangle + diagonal (36 blocks).
/// BSIZE-interleaved: one struct per group of BSIZE elements.
/// data[value_index][lane] where value_index = blockIdx*9 + mat_value (0..323),
/// lane = elem % BSIZE.  Threads with consecutive lanes access consecutive
/// addresses — perfect coalescing.
template<class real>
class GPUKMatrix
{
public:
    static constexpr int NBLOCKS = 36;
    real data[NBLOCKS * 9][BSIZE];
};

/// Block index mapping for the symmetric K matrix (upper triangle, row by row).
/// blockIdx(r, c) = 8*r - r*(r-1)/2 + (c - r) for r <= c.
enum KBlockIdx {
    KB_D0  =  0, KB_U01 =  1, KB_U02 =  2, KB_U03 =  3, KB_U04 =  4, KB_U05 =  5, KB_U06 =  6, KB_U07 =  7,
    KB_D1  =  8, KB_U12 =  9, KB_U13 = 10, KB_U14 = 11, KB_U15 = 12, KB_U16 = 13, KB_U17 = 14,
    KB_D2  = 15, KB_U23 = 16, KB_U24 = 17, KB_U25 = 18, KB_U26 = 19, KB_U27 = 20,
    KB_D3  = 21, KB_U34 = 22, KB_U35 = 23, KB_U36 = 24, KB_U37 = 25,
    KB_D4  = 26, KB_U45 = 27, KB_U46 = 28, KB_U47 = 29,
    KB_D5  = 30, KB_U56 = 31, KB_U57 = 32,
    KB_D6  = 33, KB_U67 = 34,
    KB_D7  = 35
};

/// Load a 3x3 block from the BSIZE-interleaved K matrix layout using __ldg.
/// kbase points to data[0][lane] for the current thread's lane within its group.
template<typename real>
__device__ __forceinline__
matrix3<real> loadKBlock(const real* __restrict__ kbase, int blockIdx)
{
    const real* p = kbase + blockIdx * 9 * BSIZE;
    matrix3<real> m;
    m.x.x = __ldg(p); p += BSIZE;
    m.x.y = __ldg(p); p += BSIZE;
    m.x.z = __ldg(p); p += BSIZE;
    m.y.x = __ldg(p); p += BSIZE;
    m.y.y = __ldg(p); p += BSIZE;
    m.y.z = __ldg(p); p += BSIZE;
    m.z.x = __ldg(p); p += BSIZE;
    m.z.y = __ldg(p); p += BSIZE;
    m.z.z = __ldg(p);
    return m;
}

extern "C"
{
    void HexahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, const void* elems, void* state, const void* kmatrix, void* f, const void* x);
    void HexahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, const void* elems, const void* state, const void* kmatrix, void* df, const void* dx, double kFactor);
    void HexahedronFEMForceFieldCuda3f_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
    void HexahedronFEMForceFieldCuda3f_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);
}

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
__global__ void __launch_bounds__(BLOCK_SIZE)
calcForce_kernel(
    int nbElem,
    const GPUElement<real>* __restrict__ elems,
    real* __restrict__ stateRotations,
    const GPUKMatrix<real>* __restrict__ kmatrix,
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
    const int vg = __ldg(&e->ig[li]);
    const int vh = __ldg(&e->ih[li]);
    const int vi = __ldg(&e->ii[li]);
    const int vj = __ldg(&e->ij[li]);

    // Load current positions
    CudaVec3<real> IA = loadPos(x, va);
    CudaVec3<real> IB = loadPos(x, vb);
    CudaVec3<real> IC = loadPos(x, vc);
    CudaVec3<real> ID = loadPos(x, vd);
    CudaVec3<real> IG = loadPos(x, vg);
    CudaVec3<real> IH = loadPos(x, vh);
    CudaVec3<real> II = loadPos(x, vi);
    CudaVec3<real> IJ = loadPos(x, vj);

    // Compute rotation from edge vectors
    CudaVec3<real> horizontal = IB - IA + IC - ID + IH - IG + II - IJ;
    CudaVec3<real> vertical = ID - IA + IC - IB + IJ - IG + II - IH;

    matrix3<real> Rt;
    Rt.x = horizontal;
    Rt.x *= invnorm(Rt.x);

    Rt.z = cross(horizontal, vertical);
    Rt.y = cross(Rt.z, horizontal);
    Rt.y *= invnorm(Rt.y);
    Rt.z *= invnorm(Rt.z);

    // Compute displacements: d[v] = restPos[v] - Rt * currentPos[v]
    CudaVec3<real> dA, dB, dC, dD, dG, dH, dI, dJ;

    dA.x = __ldg(&e->ax[li]) - Rt.mulX(IA);
    dA.y = __ldg(&e->ay[li]) - Rt.mulY(IA);
    dA.z = __ldg(&e->az[li]) - Rt.mulZ(IA);

    dB.x = __ldg(&e->bx[li]) - Rt.mulX(IB);
    dB.y = __ldg(&e->by[li]) - Rt.mulY(IB);
    dB.z = __ldg(&e->bz[li]) - Rt.mulZ(IB);

    dC.x = __ldg(&e->cx[li]) - Rt.mulX(IC);
    dC.y = __ldg(&e->cy[li]) - Rt.mulY(IC);
    dC.z = __ldg(&e->cz[li]) - Rt.mulZ(IC);

    dD.x = __ldg(&e->dx[li]) - Rt.mulX(ID);
    dD.y = __ldg(&e->dy[li]) - Rt.mulY(ID);
    dD.z = __ldg(&e->dz[li]) - Rt.mulZ(ID);

    dG.x = __ldg(&e->gx[li]) - Rt.mulX(IG);
    dG.y = __ldg(&e->gy[li]) - Rt.mulY(IG);
    dG.z = __ldg(&e->gz[li]) - Rt.mulZ(IG);

    dH.x = __ldg(&e->hx[li]) - Rt.mulX(IH);
    dH.y = __ldg(&e->hy[li]) - Rt.mulY(IH);
    dH.z = __ldg(&e->hz[li]) - Rt.mulZ(IH);

    dI.x = __ldg(&e->ix[li]) - Rt.mulX(II);
    dI.y = __ldg(&e->iy[li]) - Rt.mulY(II);
    dI.z = __ldg(&e->iz[li]) - Rt.mulZ(II);

    dJ.x = __ldg(&e->jx[li]) - Rt.mulX(IJ);
    dJ.y = __ldg(&e->jy[li]) - Rt.mulY(IJ);
    dJ.z = __ldg(&e->jz[li]) - Rt.mulZ(IJ);

    // Store rotation in BSIZE-interleaved format (for addDForce / getRotations)
    const int rotBase = (elem / BSIZE) * (9 * BSIZE) + li;
    Rt.writeAoS(stateRotations + rotBase, BSIZE);

    // Exploit K matrix symmetry: K[i][j] = transpose(K[j][i]).
    // Load only the 36 upper-triangle blocks (44% less bandwidth)
    // and accumulate all 8 vertex forces simultaneously.
    // kbase points to data[0][li] for this thread's lane within its BSIZE group.
    const real* kbase = &(kmatrix + (elem / BSIZE))->data[0][li];
    matrix3<real> K;

    // Diagonal contributions: f[i] = K[i][i] * d[i]
    K = loadKBlock(kbase, KB_D0); CudaVec3<real> fA = K * dA;
    K = loadKBlock(kbase, KB_D1); CudaVec3<real> fB = K * dB;
    K = loadKBlock(kbase, KB_D2); CudaVec3<real> fC = K * dC;
    K = loadKBlock(kbase, KB_D3); CudaVec3<real> fD = K * dD;
    K = loadKBlock(kbase, KB_D4); CudaVec3<real> fG = K * dG;
    K = loadKBlock(kbase, KB_D5); CudaVec3<real> fH = K * dH;
    K = loadKBlock(kbase, KB_D6); CudaVec3<real> fI = K * dI;
    K = loadKBlock(kbase, KB_D7); CudaVec3<real> fJ = K * dJ;

    // Off-diagonal pairs: K[i][j] contributes to both f[i] and f[j]
    // f[i] += K[i][j] * d[j],  f[j] += K[i][j]^T * d[i]

    // Row 0 pairs
    K = loadKBlock(kbase, KB_U01); fA += K * dB; fB += K.mulT(dA);
    K = loadKBlock(kbase, KB_U02); fA += K * dC; fC += K.mulT(dA);
    K = loadKBlock(kbase, KB_U03); fA += K * dD; fD += K.mulT(dA);
    K = loadKBlock(kbase, KB_U04); fA += K * dG; fG += K.mulT(dA);
    K = loadKBlock(kbase, KB_U05); fA += K * dH; fH += K.mulT(dA);
    K = loadKBlock(kbase, KB_U06); fA += K * dI; fI += K.mulT(dA);
    K = loadKBlock(kbase, KB_U07); fA += K * dJ; fJ += K.mulT(dA);

    // Row 1 pairs
    K = loadKBlock(kbase, KB_U12); fB += K * dC; fC += K.mulT(dB);
    K = loadKBlock(kbase, KB_U13); fB += K * dD; fD += K.mulT(dB);
    K = loadKBlock(kbase, KB_U14); fB += K * dG; fG += K.mulT(dB);
    K = loadKBlock(kbase, KB_U15); fB += K * dH; fH += K.mulT(dB);
    K = loadKBlock(kbase, KB_U16); fB += K * dI; fI += K.mulT(dB);
    K = loadKBlock(kbase, KB_U17); fB += K * dJ; fJ += K.mulT(dB);

    // Row 2 pairs
    K = loadKBlock(kbase, KB_U23); fC += K * dD; fD += K.mulT(dC);
    K = loadKBlock(kbase, KB_U24); fC += K * dG; fG += K.mulT(dC);
    K = loadKBlock(kbase, KB_U25); fC += K * dH; fH += K.mulT(dC);
    K = loadKBlock(kbase, KB_U26); fC += K * dI; fI += K.mulT(dC);
    K = loadKBlock(kbase, KB_U27); fC += K * dJ; fJ += K.mulT(dC);

    // Row 3 pairs
    K = loadKBlock(kbase, KB_U34); fD += K * dG; fG += K.mulT(dD);
    K = loadKBlock(kbase, KB_U35); fD += K * dH; fH += K.mulT(dD);
    K = loadKBlock(kbase, KB_U36); fD += K * dI; fI += K.mulT(dD);
    K = loadKBlock(kbase, KB_U37); fD += K * dJ; fJ += K.mulT(dD);

    // Row 4 pairs
    K = loadKBlock(kbase, KB_U45); fG += K * dH; fH += K.mulT(dG);
    K = loadKBlock(kbase, KB_U46); fG += K * dI; fI += K.mulT(dG);
    K = loadKBlock(kbase, KB_U47); fG += K * dJ; fJ += K.mulT(dG);

    // Row 5 pairs
    K = loadKBlock(kbase, KB_U56); fH += K * dI; fI += K.mulT(dH);
    K = loadKBlock(kbase, KB_U57); fH += K * dJ; fJ += K.mulT(dH);

    // Row 6 pairs
    K = loadKBlock(kbase, KB_U67); fI += K * dJ; fJ += K.mulT(dI);

    // Back-rotate and scatter
    atomicAddVec3(f, va, Rt.mulT(fA));
    atomicAddVec3(f, vb, Rt.mulT(fB));
    atomicAddVec3(f, vc, Rt.mulT(fC));
    atomicAddVec3(f, vd, Rt.mulT(fD));
    atomicAddVec3(f, vg, Rt.mulT(fG));
    atomicAddVec3(f, vh, Rt.mulT(fH));
    atomicAddVec3(f, vi, Rt.mulT(fI));
    atomicAddVec3(f, vj, Rt.mulT(fJ));
}

template<typename real>
__global__ void __launch_bounds__(BLOCK_SIZE)
calcDForce_kernel(
    int nbElem,
    const GPUElement<real>* __restrict__ elems,
    const real* __restrict__ stateRotations,
    const GPUKMatrix<real>* __restrict__ kmatrix,
    const CudaVec3<real>* __restrict__ dx,
    CudaVec3<real>* df,
    real fact)
{
    const int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= nbElem) return;

    // Access BSIZE-blocked element data
    const GPUElement<real>* e = elems + (elem / BSIZE);
    const int li = elem % BSIZE;

    // Read rotation from state buffer (written by prior calcForce call)
    matrix3<real> Rt;
    const int rotBase = (elem / BSIZE) * (9 * BSIZE) + li;
    Rt.readAoS(stateRotations + rotBase, BSIZE);

    // Read vertex indices via __ldg
    const int va = __ldg(&e->ia[li]);
    const int vb = __ldg(&e->ib[li]);
    const int vc = __ldg(&e->ic[li]);
    const int vd = __ldg(&e->id[li]);
    const int vg = __ldg(&e->ig[li]);
    const int vh = __ldg(&e->ih[li]);
    const int vi = __ldg(&e->ii[li]);
    const int vj = __ldg(&e->ij[li]);

    // Rotate displacements into element frame
    CudaVec3<real> dA = Rt * loadPos(dx, va);
    CudaVec3<real> dB = Rt * loadPos(dx, vb);
    CudaVec3<real> dC = Rt * loadPos(dx, vc);
    CudaVec3<real> dD = Rt * loadPos(dx, vd);
    CudaVec3<real> dG = Rt * loadPos(dx, vg);
    CudaVec3<real> dH = Rt * loadPos(dx, vh);
    CudaVec3<real> dI = Rt * loadPos(dx, vi);
    CudaVec3<real> dJ = Rt * loadPos(dx, vj);

    // Symmetric K*d computation with pre-negated factor
    const real* kbase = &(kmatrix + (elem / BSIZE))->data[0][li];
    const real nfact = -fact;
    matrix3<real> K;

    // Diagonal contributions
    K = loadKBlock(kbase, KB_D0); CudaVec3<real> fA = K * dA;
    K = loadKBlock(kbase, KB_D1); CudaVec3<real> fB = K * dB;
    K = loadKBlock(kbase, KB_D2); CudaVec3<real> fC = K * dC;
    K = loadKBlock(kbase, KB_D3); CudaVec3<real> fD = K * dD;
    K = loadKBlock(kbase, KB_D4); CudaVec3<real> fG = K * dG;
    K = loadKBlock(kbase, KB_D5); CudaVec3<real> fH = K * dH;
    K = loadKBlock(kbase, KB_D6); CudaVec3<real> fI = K * dI;
    K = loadKBlock(kbase, KB_D7); CudaVec3<real> fJ = K * dJ;

    // Off-diagonal pairs (symmetric)
    K = loadKBlock(kbase, KB_U01); fA += K * dB; fB += K.mulT(dA);
    K = loadKBlock(kbase, KB_U02); fA += K * dC; fC += K.mulT(dA);
    K = loadKBlock(kbase, KB_U03); fA += K * dD; fD += K.mulT(dA);
    K = loadKBlock(kbase, KB_U04); fA += K * dG; fG += K.mulT(dA);
    K = loadKBlock(kbase, KB_U05); fA += K * dH; fH += K.mulT(dA);
    K = loadKBlock(kbase, KB_U06); fA += K * dI; fI += K.mulT(dA);
    K = loadKBlock(kbase, KB_U07); fA += K * dJ; fJ += K.mulT(dA);

    K = loadKBlock(kbase, KB_U12); fB += K * dC; fC += K.mulT(dB);
    K = loadKBlock(kbase, KB_U13); fB += K * dD; fD += K.mulT(dB);
    K = loadKBlock(kbase, KB_U14); fB += K * dG; fG += K.mulT(dB);
    K = loadKBlock(kbase, KB_U15); fB += K * dH; fH += K.mulT(dB);
    K = loadKBlock(kbase, KB_U16); fB += K * dI; fI += K.mulT(dB);
    K = loadKBlock(kbase, KB_U17); fB += K * dJ; fJ += K.mulT(dB);

    K = loadKBlock(kbase, KB_U23); fC += K * dD; fD += K.mulT(dC);
    K = loadKBlock(kbase, KB_U24); fC += K * dG; fG += K.mulT(dC);
    K = loadKBlock(kbase, KB_U25); fC += K * dH; fH += K.mulT(dC);
    K = loadKBlock(kbase, KB_U26); fC += K * dI; fI += K.mulT(dC);
    K = loadKBlock(kbase, KB_U27); fC += K * dJ; fJ += K.mulT(dC);

    K = loadKBlock(kbase, KB_U34); fD += K * dG; fG += K.mulT(dD);
    K = loadKBlock(kbase, KB_U35); fD += K * dH; fH += K.mulT(dD);
    K = loadKBlock(kbase, KB_U36); fD += K * dI; fI += K.mulT(dD);
    K = loadKBlock(kbase, KB_U37); fD += K * dJ; fJ += K.mulT(dD);

    K = loadKBlock(kbase, KB_U45); fG += K * dH; fH += K.mulT(dG);
    K = loadKBlock(kbase, KB_U46); fG += K * dI; fI += K.mulT(dG);
    K = loadKBlock(kbase, KB_U47); fG += K * dJ; fJ += K.mulT(dG);

    K = loadKBlock(kbase, KB_U56); fH += K * dI; fI += K.mulT(dH);
    K = loadKBlock(kbase, KB_U57); fH += K * dJ; fJ += K.mulT(dH);

    K = loadKBlock(kbase, KB_U67); fI += K * dJ; fJ += K.mulT(dI);

    // Back-rotate, scale by -kFactor, and scatter
    atomicAddVec3(df, va, Rt.mulT(fA) * nfact);
    atomicAddVec3(df, vb, Rt.mulT(fB) * nfact);
    atomicAddVec3(df, vc, Rt.mulT(fC) * nfact);
    atomicAddVec3(df, vd, Rt.mulT(fD) * nfact);
    atomicAddVec3(df, vg, Rt.mulT(fG) * nfact);
    atomicAddVec3(df, vh, Rt.mulT(fH) * nfact);
    atomicAddVec3(df, vi, Rt.mulT(fI) * nfact);
    atomicAddVec3(df, vj, Rt.mulT(fJ) * nfact);
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

void HexahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, const void* elems, void* state, const void* kmatrix, void* f, const void* x)
{
    dim3 grid((nbElem + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    calcForce_kernel<float><<<grid, threads>>>(
        nbElem,
        (const GPUElement<float>*)elems,
        (float*)state,
        (const GPUKMatrix<float>*)kmatrix,
        (const CudaVec3<float>*)x,
        (CudaVec3<float>*)f);
    mycudaDebugError("calcForce_kernel<float>");
}

void HexahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, const void* elems, const void* state, const void* kmatrix, void* df, const void* dx, double kFactor)
{
    dim3 grid((nbElem + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    calcDForce_kernel<float><<<grid, threads>>>(
        nbElem,
        (const GPUElement<float>*)elems,
        (const float*)state,
        (const GPUKMatrix<float>*)kmatrix,
        (const CudaVec3<float>*)dx,
        (CudaVec3<float>*)df,
        (float)kFactor);
    mycudaDebugError("calcDForce_kernel<float>");
}

void HexahedronFEMForceFieldCuda3f_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
{
    dim3 grid((nbVertex + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    getRotations_kernel<float><<<grid, threads>>>(nbVertex, (const float*)initState, (const float*)state, (const int*)rotationIdx, (float*)rotations);
    mycudaDebugError("getRotations_kernel<float>");
}

void HexahedronFEMForceFieldCuda3f_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations)
{
    dim3 grid((nbElem + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    getElementRotations_kernel<float><<<grid, threads>>>(nbElem, (const float*)rotationsAos, (float*)rotations);
    mycudaDebugError("getElementRotations_kernel<float>");
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
