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

extern "C"
{
    void CudaVisualModelCuda3f_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

    void CudaVisualModelCuda3f1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

    // New atomic-based normal computation (simpler, no velems needed)
    void CudaVisualModelCuda3f_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex,
                                                  const void* triangles, const void* quads, void* vnormals, const void* x);
    void CudaVisualModelCuda3f1_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex,
                                                   const void* triangles, const void* quads, void* vnormals, const void* x);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void CudaVisualModelCuda3d_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

    void CudaVisualModelCuda3d1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

    void CudaVisualModelCuda3d_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex,
                                                  const void* triangles, const void* quads, void* vnormals, const void* x);
    void CudaVisualModelCuda3d1_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex,
                                                   const void* triangles, const void* quads, void* vnormals, const void* x);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

// #define USE_TEXTURE false
#ifdef USE_TEXTURE
#undef USE_TEXTURE
#endif

// no texture is used unless this template is specialized
template<typename real, class TIn>
class CudaCudaVisualModelTextures
{
public:

    static __host__ void setX(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }

    static __host__ void setN(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getN(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }
};


#ifdef USE_TEXTURE

static texture<float,1,cudaReadModeElementType> tex_3f_x;
static texture<float,1,cudaReadModeElementType> tex_3f_n;

template<>
class CudaCudaVisualModelTextures<float, CudaVec3<float> >
{
public:
    typedef float real;
    typedef CudaVec3<real> TIn;

    static __host__ void setX(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_x, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
        int i3 = i * 3;
        float x1 = tex1Dfetch(tex_3f_x, i3);
        float x2 = tex1Dfetch(tex_3f_x, i3+1);
        float x3 = tex1Dfetch(tex_3f_x, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }

    static __host__ void setN(const void* n)
    {
        static const void* cur = NULL;
        if (n!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_n, n);
            cur = n;
        }
    }

    static __inline__ __device__ CudaVec3<real> getN(int i, const TIn* n)
    {
        int i3 = i * 3;
        float x1 = tex1Dfetch(tex_3f_n, i3);
        float x2 = tex1Dfetch(tex_3f_n, i3+1);
        float x3 = tex1Dfetch(tex_3f_n, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }
};

static texture<float4,1,cudaReadModeElementType> tex_3f1_x;
static texture<float4,1,cudaReadModeElementType> tex_3f1_n;

template<>
class CudaCudaVisualModelTextures<float, CudaVec4<float> >
{
public:
    typedef float real;
    typedef CudaVec4<real> TIn;

    static __host__ void setX(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_x, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_x, i));
    }

    static __host__ void setN(const void* n)
    {
        static const void* cur = NULL;
        if (n!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_n, n);
            cur = n;
        }
    }

    static __inline__ __device__ CudaVec3<real> getN(int i, const TIn* n)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_n, i));
    }
};

#endif

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t_calcTNormals_kernel(int nbElem, const int* __restrict__ elems, real* __restrict__ fnormals, const TIn* __restrict__ x)
{
    const int index0 = blockIdx.x * BSIZE;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;
    const int index3 = index1 * 3;
    const int iext = index0 * 3 + index1;

    __shared__ union
    {
        int itemp[3*BSIZE];
        real rtemp[3*BSIZE];
    } s;

    s.itemp[index1          ] = elems[iext          ];
    s.itemp[index1 +   BSIZE] = elems[iext +   BSIZE];
    s.itemp[index1 + 2*BSIZE] = elems[iext + 2*BSIZE];

    __syncthreads();

    CudaVec3<real> N = CudaVec3<real>::make(0, 0, 0);
    if (index < nbElem)
    {
        CudaVec3<real> A = CudaCudaVisualModelTextures<real,TIn>::getX(s.itemp[index3    ], x);
        CudaVec3<real> B = CudaCudaVisualModelTextures<real,TIn>::getX(s.itemp[index3 + 1], x);
        CudaVec3<real> C = CudaCudaVisualModelTextures<real,TIn>::getX(s.itemp[index3 + 2], x);
        B -= A;
        C -= A;
        N = cross(B, C);
        N *= invnorm(N);
    }

    if (sizeof(real) != sizeof(int)) __syncthreads();

    s.rtemp[index3    ] = N.x;
    s.rtemp[index3 + 1] = N.y;
    s.rtemp[index3 + 2] = N.z;

    __syncthreads();

    fnormals[iext          ] = s.rtemp[index1          ];
    fnormals[iext +   BSIZE] = s.rtemp[index1 +   BSIZE];
    fnormals[iext + 2*BSIZE] = s.rtemp[index1 + 2*BSIZE];
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t1_calcTNormals_kernel(int nbElem, const int* __restrict__ elems, CudaVec4<real>* __restrict__ fnormals, const TIn* __restrict__ x)
{
    const int index0 = blockIdx.x * BSIZE;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;
    const int index3 = index1 * 3;
    const int iext = index0 * 3 + index1;

    __shared__ int itemp[3*BSIZE];

    itemp[index1          ] = elems[iext          ];
    itemp[index1 +   BSIZE] = elems[iext +   BSIZE];
    itemp[index1 + 2*BSIZE] = elems[iext + 2*BSIZE];

    __syncthreads();

    CudaVec3<real> N = CudaVec3<real>::make(0, 0, 0);
    if (index < nbElem)
    {
        CudaVec3<real> A = CudaCudaVisualModelTextures<real,TIn>::getX(itemp[index3    ], x);
        CudaVec3<real> B = CudaCudaVisualModelTextures<real,TIn>::getX(itemp[index3 + 1], x);
        CudaVec3<real> C = CudaCudaVisualModelTextures<real,TIn>::getX(itemp[index3 + 2], x);
        B -= A;
        C -= A;
        N = cross(B, C);
        N *= invnorm(N);
    }

    fnormals[index] = CudaVec4<real>::make(N, 0.0f);
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t_calcQNormals_kernel(int nbElem, const int4* __restrict__ elems, real* __restrict__ fnormals, const TIn* __restrict__ x)
{
    const int index0 = blockIdx.x * BSIZE;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;
    const int index3 = index1 * 3;
    const int iext = index0 * 3 + index1;

    __shared__ real rtemp[3*BSIZE];

    CudaVec3<real> N = CudaVec3<real>::make(0, 0, 0);
    if (index < nbElem)
    {
        int4 itemp = elems[index];
        CudaVec3<real> A = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.x, x);
        CudaVec3<real> B = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.y, x);
        CudaVec3<real> C = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.z, x);
        CudaVec3<real> D = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.w, x);
        C -= A;
        D -= B;
        N = cross(C, D);
        N *= invnorm(N);
    }

    rtemp[index3    ] = N.x;
    rtemp[index3 + 1] = N.y;
    rtemp[index3 + 2] = N.z;

    __syncthreads();

    fnormals[iext          ] = rtemp[index1          ];
    fnormals[iext +   BSIZE] = rtemp[index1 +   BSIZE];
    fnormals[iext + 2*BSIZE] = rtemp[index1 + 2*BSIZE];
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t1_calcQNormals_kernel(int nbElem, const int4* __restrict__ elems, CudaVec4<real>* __restrict__ fnormals, const TIn* __restrict__ x)
{
    const int index0 = blockIdx.x * BSIZE;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;

    CudaVec3<real> N = CudaVec3<real>::make(0, 0, 0);
    if (index < nbElem)
    {
        int4 itemp = elems[index];
        CudaVec3<real> A = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.x, x);
        CudaVec3<real> B = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.y, x);
        CudaVec3<real> C = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.z, x);
        CudaVec3<real> D = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.w, x);
        C -= A;
        D -= B;
        N = cross(C, D);
        N *= invnorm(N);
    }

    fnormals[index] = CudaVec4<real>::make(N, 0.0f);
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t_calcVNormals_kernel(int nbVertex, unsigned int nbElemPerVertex, const int* __restrict__ velems, real* __restrict__ vnormals, const TIn* __restrict__ fnormals)
{
    const int index0 = blockIdx.x * BSIZE;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;
    const int index3 = index1 * 3;
    const int iext = index0 * 3 + index1;

    __shared__ real temp[3*BSIZE];

    CudaVec3<real> n = CudaVec3<real>::make(0.0f, 0.0f, 0.0f);

    if (index < nbVertex)
    {
        const int* vptr = velems + index0 * nbElemPerVertex + index1;
        for (unsigned int s = 0; s < nbElemPerVertex; s++)
        {
            int i = vptr[s * BSIZE] - 1;
            if (i >= 0)
            {
                n += CudaCudaVisualModelTextures<real,TIn>::getN(i, fnormals);
            }
        }
        real invn = invnorm(n);
        if (invn < 100000.0f)
            n *= invn;
    }

    temp[index3    ] = n.x;
    temp[index3 + 1] = n.y;
    temp[index3 + 2] = n.z;

    __syncthreads();

    vnormals[iext          ] = temp[index1          ];
    vnormals[iext +   BSIZE] = temp[index1 +   BSIZE];
    vnormals[iext + 2*BSIZE] = temp[index1 + 2*BSIZE];
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t1_calcVNormals_kernel(int nbVertex, unsigned int nbElemPerVertex, const int* __restrict__ velems, CudaVec4<real>* __restrict__ vnormals, const TIn* __restrict__ fnormals)
{
    const int index0 = blockIdx.x * BSIZE;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;

    CudaVec3<real> n = CudaVec3<real>::make(0.0f, 0.0f, 0.0f);

    if (index < nbVertex)
    {
        const int* vptr = velems + index0 * nbElemPerVertex + index1;
        for (unsigned int s = 0; s < nbElemPerVertex; s++)
        {
            int i = vptr[s * BSIZE] - 1;
            if (i >= 0)
            {
                n += CudaCudaVisualModelTextures<real,TIn>::getN(i, fnormals);
            }
        }
        real invn = invnorm(n);
        if (invn < 100000.0f)
            n *= invn;
    }
    vnormals[index] = CudaVec4<real>::make(n, 0.0f);
}

//////////////////////
// Atomic-based normal computation kernels
//////////////////////

// Scatter triangle normals to vertices using atomics (Vec3 layout)
template<typename real, class TIn>
__global__ void CudaVisualModel_scatterTriangleNormals_kernel(
    int nbTriangles,
    const int* __restrict__ triangles,
    real* __restrict__ vnormals,
    const TIn* __restrict__ x)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nbTriangles) return;

    // Load triangle indices
    const int i0 = triangles[tid * 3 + 0];
    const int i1 = triangles[tid * 3 + 1];
    const int i2 = triangles[tid * 3 + 2];

    // Load positions
    CudaVec3<real> A = CudaCudaVisualModelTextures<real, TIn>::getX(i0, x);
    CudaVec3<real> B = CudaCudaVisualModelTextures<real, TIn>::getX(i1, x);
    CudaVec3<real> C = CudaCudaVisualModelTextures<real, TIn>::getX(i2, x);

    // Compute face normal (unnormalized - will normalize later)
    B -= A;
    C -= A;
    CudaVec3<real> N = cross(B, C);

    // Atomically add to each vertex's normal
    atomicAdd(&vnormals[i0 * 3 + 0], N.x);
    atomicAdd(&vnormals[i0 * 3 + 1], N.y);
    atomicAdd(&vnormals[i0 * 3 + 2], N.z);

    atomicAdd(&vnormals[i1 * 3 + 0], N.x);
    atomicAdd(&vnormals[i1 * 3 + 1], N.y);
    atomicAdd(&vnormals[i1 * 3 + 2], N.z);

    atomicAdd(&vnormals[i2 * 3 + 0], N.x);
    atomicAdd(&vnormals[i2 * 3 + 1], N.y);
    atomicAdd(&vnormals[i2 * 3 + 2], N.z);
}

// Scatter quad normals to vertices using atomics (Vec3 layout)
template<typename real, class TIn>
__global__ void CudaVisualModel_scatterQuadNormals_kernel(
    int nbQuads,
    const int* __restrict__ quads,
    real* __restrict__ vnormals,
    const TIn* __restrict__ x)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nbQuads) return;

    // Load quad indices
    const int i0 = quads[tid * 4 + 0];
    const int i1 = quads[tid * 4 + 1];
    const int i2 = quads[tid * 4 + 2];
    const int i3 = quads[tid * 4 + 3];

    // Load positions
    CudaVec3<real> A = CudaCudaVisualModelTextures<real, TIn>::getX(i0, x);
    CudaVec3<real> B = CudaCudaVisualModelTextures<real, TIn>::getX(i1, x);
    CudaVec3<real> C = CudaCudaVisualModelTextures<real, TIn>::getX(i2, x);
    CudaVec3<real> D = CudaCudaVisualModelTextures<real, TIn>::getX(i3, x);

    // Compute face normal using diagonals
    C -= A;
    D -= B;
    CudaVec3<real> N = cross(C, D);

    // Atomically add to each vertex's normal
    atomicAdd(&vnormals[i0 * 3 + 0], N.x);
    atomicAdd(&vnormals[i0 * 3 + 1], N.y);
    atomicAdd(&vnormals[i0 * 3 + 2], N.z);

    atomicAdd(&vnormals[i1 * 3 + 0], N.x);
    atomicAdd(&vnormals[i1 * 3 + 1], N.y);
    atomicAdd(&vnormals[i1 * 3 + 2], N.z);

    atomicAdd(&vnormals[i2 * 3 + 0], N.x);
    atomicAdd(&vnormals[i2 * 3 + 1], N.y);
    atomicAdd(&vnormals[i2 * 3 + 2], N.z);

    atomicAdd(&vnormals[i3 * 3 + 0], N.x);
    atomicAdd(&vnormals[i3 * 3 + 1], N.y);
    atomicAdd(&vnormals[i3 * 3 + 2], N.z);
}

// Normalize accumulated normals (Vec3 layout)
template<typename real>
__global__ void CudaVisualModel_normalizeNormals_kernel(int nbVertex, real* __restrict__ vnormals)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nbVertex) return;

    const int idx = tid * 3;
    real nx = vnormals[idx + 0];
    real ny = vnormals[idx + 1];
    real nz = vnormals[idx + 2];

    real n2 = nx * nx + ny * ny + nz * nz;
    real invLen = (n2 > 0) ? rsqrt(n2) : 0;

    vnormals[idx + 0] = nx * invLen;
    vnormals[idx + 1] = ny * invLen;
    vnormals[idx + 2] = nz * invLen;
}

// Scatter triangle normals to vertices using atomics (Vec4 layout)
template<typename real, class TIn>
__global__ void CudaVisualModel_scatterTriangleNormals4_kernel(
    int nbTriangles,
    const int* __restrict__ triangles,
    CudaVec4<real>* __restrict__ vnormals,
    const TIn* __restrict__ x)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nbTriangles) return;

    const int i0 = triangles[tid * 3 + 0];
    const int i1 = triangles[tid * 3 + 1];
    const int i2 = triangles[tid * 3 + 2];

    CudaVec3<real> A = CudaCudaVisualModelTextures<real, TIn>::getX(i0, x);
    CudaVec3<real> B = CudaCudaVisualModelTextures<real, TIn>::getX(i1, x);
    CudaVec3<real> C = CudaCudaVisualModelTextures<real, TIn>::getX(i2, x);

    B -= A;
    C -= A;
    CudaVec3<real> N = cross(B, C);

    // For Vec4 layout, we need to cast to float* for atomicAdd
    real* vn = (real*)vnormals;
    atomicAdd(&vn[i0 * 4 + 0], N.x);
    atomicAdd(&vn[i0 * 4 + 1], N.y);
    atomicAdd(&vn[i0 * 4 + 2], N.z);

    atomicAdd(&vn[i1 * 4 + 0], N.x);
    atomicAdd(&vn[i1 * 4 + 1], N.y);
    atomicAdd(&vn[i1 * 4 + 2], N.z);

    atomicAdd(&vn[i2 * 4 + 0], N.x);
    atomicAdd(&vn[i2 * 4 + 1], N.y);
    atomicAdd(&vn[i2 * 4 + 2], N.z);
}

// Scatter quad normals to vertices using atomics (Vec4 layout)
template<typename real, class TIn>
__global__ void CudaVisualModel_scatterQuadNormals4_kernel(
    int nbQuads,
    const int* __restrict__ quads,
    CudaVec4<real>* __restrict__ vnormals,
    const TIn* __restrict__ x)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nbQuads) return;

    const int i0 = quads[tid * 4 + 0];
    const int i1 = quads[tid * 4 + 1];
    const int i2 = quads[tid * 4 + 2];
    const int i3 = quads[tid * 4 + 3];

    CudaVec3<real> A = CudaCudaVisualModelTextures<real, TIn>::getX(i0, x);
    CudaVec3<real> B = CudaCudaVisualModelTextures<real, TIn>::getX(i1, x);
    CudaVec3<real> C = CudaCudaVisualModelTextures<real, TIn>::getX(i2, x);
    CudaVec3<real> D = CudaCudaVisualModelTextures<real, TIn>::getX(i3, x);

    C -= A;
    D -= B;
    CudaVec3<real> N = cross(C, D);

    real* vn = (real*)vnormals;
    atomicAdd(&vn[i0 * 4 + 0], N.x);
    atomicAdd(&vn[i0 * 4 + 1], N.y);
    atomicAdd(&vn[i0 * 4 + 2], N.z);

    atomicAdd(&vn[i1 * 4 + 0], N.x);
    atomicAdd(&vn[i1 * 4 + 1], N.y);
    atomicAdd(&vn[i1 * 4 + 2], N.z);

    atomicAdd(&vn[i2 * 4 + 0], N.x);
    atomicAdd(&vn[i2 * 4 + 1], N.y);
    atomicAdd(&vn[i2 * 4 + 2], N.z);

    atomicAdd(&vn[i3 * 4 + 0], N.x);
    atomicAdd(&vn[i3 * 4 + 1], N.y);
    atomicAdd(&vn[i3 * 4 + 2], N.z);
}

// Normalize accumulated normals (Vec4 layout)
template<typename real>
__global__ void CudaVisualModel_normalizeNormals4_kernel(int nbVertex, CudaVec4<real>* __restrict__ vnormals)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nbVertex) return;

    CudaVec4<real> n = vnormals[tid];
    real n2 = n.x * n.x + n.y * n.y + n.z * n.z;
    real invLen = (n2 > 0) ? rsqrt(n2) : 0;

    vnormals[tid] = CudaVec4<real>::make(n.x * invLen, n.y * invLen, n.z * invLen, 0);
}

//////////////////////
// CPU-side methods //
//////////////////////

void CudaVisualModelCuda3f_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<float,CudaVec3<float> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcTNormals_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (float*)fnormals, (const CudaVec3<float>*)x); mycudaDebugError("CudaVisualModelCuda3t_calcTNormals_kernel<float, CudaVec3<float> >");}
}

void CudaVisualModelCuda3f_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<float,CudaVec3<float> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcQNormals_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const int4*)elems, (float*)fnormals, (const CudaVec3<float>*)x); mycudaDebugError("CudaVisualModelCuda3t_calcQNormals_kernel<float, CudaVec3<float> >");}
}

void CudaVisualModelCuda3f_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcVNormals_kernel<float, CudaVec3<float> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (float*)vnormals, (const CudaVec3<float>*)fnormals); mycudaDebugError("CudaVisualModelCuda3t_calcVNormals_kernel<float, CudaVec3<float> >");}
}

void CudaVisualModelCuda3f1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<float,CudaVec4<float> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcTNormals_kernel<float, CudaVec4<float> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (CudaVec4<float>*)fnormals, (const CudaVec4<float>*)x); mycudaDebugError("CudaVisualModelCuda3t1_calcTNormals_kernel<float, CudaVec4<float> >");}
}

void CudaVisualModelCuda3f1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<float,CudaVec4<float> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcQNormals_kernel<float, CudaVec4<float> ><<< grid1, threads1 >>>(nbElem, (const int4*)elems, (CudaVec4<float>*)fnormals, (const CudaVec4<float>*)x); mycudaDebugError("CudaVisualModelCuda3t1_calcQNormals_kernel<float, CudaVec4<float> >");}
}

void CudaVisualModelCuda3f1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcVNormals_kernel<float, CudaVec4<float> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (CudaVec4<float>*)vnormals, (const CudaVec4<float>*)fnormals); mycudaDebugError("CudaVisualModelCuda3t1_calcVNormals_kernel<float, CudaVec4<float> >");}
}

// Atomic-based normal computation for Vec3 layout
void CudaVisualModelCuda3f_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex,
                                              const void* triangles, const void* quads, void* vnormals, const void* x)
{
    // Zero-initialize normals
    cudaMemsetAsync(vnormals, 0, nbVertex * 3 * sizeof(float));

    dim3 threads(BSIZE, 1);

    // Scatter triangle normals
    if (nbTriangles > 0)
    {
        dim3 gridT((nbTriangles + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_scatterTriangleNormals_kernel<float, CudaVec3<float> ><<<gridT, threads>>>(
            nbTriangles, (const int*)triangles, (float*)vnormals, (const CudaVec3<float>*)x);
        mycudaDebugError("CudaVisualModel_scatterTriangleNormals_kernel");
    }

    // Scatter quad normals
    if (nbQuads > 0)
    {
        dim3 gridQ((nbQuads + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_scatterQuadNormals_kernel<float, CudaVec3<float> ><<<gridQ, threads>>>(
            nbQuads, (const int*)quads, (float*)vnormals, (const CudaVec3<float>*)x);
        mycudaDebugError("CudaVisualModel_scatterQuadNormals_kernel");
    }

    // Normalize
    if (nbVertex > 0)
    {
        dim3 gridN((nbVertex + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_normalizeNormals_kernel<float><<<gridN, threads>>>(nbVertex, (float*)vnormals);
        mycudaDebugError("CudaVisualModel_normalizeNormals_kernel");
    }
}

// Atomic-based normal computation for Vec4 layout
void CudaVisualModelCuda3f1_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex,
                                               const void* triangles, const void* quads, void* vnormals, const void* x)
{
    // Zero-initialize normals
    cudaMemsetAsync(vnormals, 0, nbVertex * 4 * sizeof(float));

    dim3 threads(BSIZE, 1);

    // Scatter triangle normals
    if (nbTriangles > 0)
    {
        dim3 gridT((nbTriangles + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_scatterTriangleNormals4_kernel<float, CudaVec4<float> ><<<gridT, threads>>>(
            nbTriangles, (const int*)triangles, (CudaVec4<float>*)vnormals, (const CudaVec4<float>*)x);
        mycudaDebugError("CudaVisualModel_scatterTriangleNormals4_kernel");
    }

    // Scatter quad normals
    if (nbQuads > 0)
    {
        dim3 gridQ((nbQuads + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_scatterQuadNormals4_kernel<float, CudaVec4<float> ><<<gridQ, threads>>>(
            nbQuads, (const int*)quads, (CudaVec4<float>*)vnormals, (const CudaVec4<float>*)x);
        mycudaDebugError("CudaVisualModel_scatterQuadNormals4_kernel");
    }

    // Normalize
    if (nbVertex > 0)
    {
        dim3 gridN((nbVertex + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_normalizeNormals4_kernel<float><<<gridN, threads>>>(nbVertex, (CudaVec4<float>*)vnormals);
        mycudaDebugError("CudaVisualModel_normalizeNormals4_kernel");
    }
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void CudaVisualModelCuda3d_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<double,CudaVec3<double> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcTNormals_kernel<double, CudaVec3<double> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (double*)fnormals, (const CudaVec3<double>*)x); mycudaDebugError("CudaVisualModelCuda3t_calcTNormals_kernel<double, CudaVec3<double> >");}
}

void CudaVisualModelCuda3d_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<double,CudaVec3<double> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcQNormals_kernel<double, CudaVec3<double> ><<< grid1, threads1 >>>(nbElem, (const int4*)elems, (double*)fnormals, (const CudaVec3<double>*)x); mycudaDebugError("CudaVisualModelCuda3t_calcQNormals_kernel<double, CudaVec3<double> >");}
}

void CudaVisualModelCuda3d_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcVNormals_kernel<double, CudaVec3<double> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (double*)vnormals, (const CudaVec3<double>*)fnormals); mycudaDebugError("CudaVisualModelCuda3t_calcVNormals_kernel<double, CudaVec3<double> >");}
}

void CudaVisualModelCuda3d1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<double,CudaVec4<double> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcTNormals_kernel<double, CudaVec4<double> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (CudaVec4<double>*)fnormals, (const CudaVec4<double>*)x); mycudaDebugError("CudaVisualModelCuda3t1_calcTNormals_kernel<double, CudaVec4<double> >");}
}

void CudaVisualModelCuda3d1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<double,CudaVec4<double> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcQNormals_kernel<double, CudaVec4<double> ><<< grid1, threads1 >>>(nbElem, (const int4*)elems, (CudaVec4<double>*)fnormals, (const CudaVec4<double>*)x); mycudaDebugError("CudaVisualModelCuda3t1_calcQNormals_kernel<double, CudaVec4<double> >");}
}

void CudaVisualModelCuda3d1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcVNormals_kernel<double, CudaVec4<double> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (CudaVec4<double>*)vnormals, (const CudaVec4<double>*)fnormals); mycudaDebugError("CudaVisualModelCuda3t1_calcVNormals_kernel<double, CudaVec4<double> >");}
}

// Atomic-based normal computation for Vec3 layout (double)
void CudaVisualModelCuda3d_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex,
                                              const void* triangles, const void* quads, void* vnormals, const void* x)
{
    cudaMemsetAsync(vnormals, 0, nbVertex * 3 * sizeof(double));

    dim3 threads(BSIZE, 1);

    if (nbTriangles > 0)
    {
        dim3 gridT((nbTriangles + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_scatterTriangleNormals_kernel<double, CudaVec3<double> ><<<gridT, threads>>>(
            nbTriangles, (const int*)triangles, (double*)vnormals, (const CudaVec3<double>*)x);
        mycudaDebugError("CudaVisualModel_scatterTriangleNormals_kernel<double>");
    }

    if (nbQuads > 0)
    {
        dim3 gridQ((nbQuads + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_scatterQuadNormals_kernel<double, CudaVec3<double> ><<<gridQ, threads>>>(
            nbQuads, (const int*)quads, (double*)vnormals, (const CudaVec3<double>*)x);
        mycudaDebugError("CudaVisualModel_scatterQuadNormals_kernel<double>");
    }

    if (nbVertex > 0)
    {
        dim3 gridN((nbVertex + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_normalizeNormals_kernel<double><<<gridN, threads>>>(nbVertex, (double*)vnormals);
        mycudaDebugError("CudaVisualModel_normalizeNormals_kernel<double>");
    }
}

// Atomic-based normal computation for Vec4 layout (double)
void CudaVisualModelCuda3d1_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex,
                                               const void* triangles, const void* quads, void* vnormals, const void* x)
{
    cudaMemsetAsync(vnormals, 0, nbVertex * 4 * sizeof(double));

    dim3 threads(BSIZE, 1);

    if (nbTriangles > 0)
    {
        dim3 gridT((nbTriangles + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_scatterTriangleNormals4_kernel<double, CudaVec4<double> ><<<gridT, threads>>>(
            nbTriangles, (const int*)triangles, (CudaVec4<double>*)vnormals, (const CudaVec4<double>*)x);
        mycudaDebugError("CudaVisualModel_scatterTriangleNormals4_kernel<double>");
    }

    if (nbQuads > 0)
    {
        dim3 gridQ((nbQuads + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_scatterQuadNormals4_kernel<double, CudaVec4<double> ><<<gridQ, threads>>>(
            nbQuads, (const int*)quads, (CudaVec4<double>*)vnormals, (const CudaVec4<double>*)x);
        mycudaDebugError("CudaVisualModel_scatterQuadNormals4_kernel<double>");
    }

    if (nbVertex > 0)
    {
        dim3 gridN((nbVertex + BSIZE - 1) / BSIZE, 1);
        CudaVisualModel_normalizeNormals4_kernel<double><<<gridN, threads>>>(nbVertex, (CudaVec4<double>*)vnormals);
        mycudaDebugError("CudaVisualModel_normalizeNormals4_kernel<double>");
    }
}

#endif // SOFA_GPU_CUDA_DOUBLE
