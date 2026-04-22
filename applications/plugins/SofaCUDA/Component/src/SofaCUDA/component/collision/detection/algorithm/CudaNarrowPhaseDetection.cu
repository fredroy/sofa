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

enum { NARROW_BSIZE = 64 };

struct NarrowPhaseTestEntry
{
    int type;
    int elem1;
    int elem2;
    int flags;
};

enum NarrowPhaseTestType
{
    TEST_TRIANGLE_POINT           = 0,
    TEST_LINE_POINT               = 1,
    TEST_LINE_LINE                = 2,
    TEST_POINT_POINT              = 3,
    TEST_TRIANGLE_TRIVERTEX       = 4,
    TEST_TRIANGLE_TRIVERTEX_SWAP  = 5,
    TEST_TRIANGLE_EDGEVERTEX      = 6,
    TEST_TRIANGLE_EDGEVERTEX_SWAP = 7,
};

struct NarrowPhaseResult
{
    CudaVec3<float> point0;
    CudaVec3<float> point1;
    CudaVec3<float> normal;
    float distance;
    int elem1;
    int elem2;
    int valid;
    int _pad;
};

extern "C"
{
    void CudaNarrowPhaseDetection_doTests(
        unsigned int nbTests,
        const void* tests,
        const void* positions1,
        const void* positions2,
        const void* triangles1,
        const void* edges1,
        const void* triangles2,
        const void* edges2,
        float alarmDist2,
        void* results);
}

__device__ CudaVec3<float> computeTriNormal(CudaVec3<float> p1, CudaVec3<float> p2, CudaVec3<float> p3)
{
    CudaVec3<float> ab = p2 - p1;
    CudaVec3<float> ac = p3 - p1;
    CudaVec3<float> n = cross(ab, ac);
    float len = norm(n);
    if (len > 1e-15f) n = n * (1.0f / len);
    return n;
}

__device__ int doIntersectionTrianglePoint_device(
    float dist2,
    int flags,
    CudaVec3<float> p1, CudaVec3<float> p2, CudaVec3<float> p3,
    CudaVec3<float> triN,
    CudaVec3<float> q,
    CudaVec3<float>& contactP,
    CudaVec3<float>& contactQ,
    CudaVec3<float>& contactNormal,
    float& contactDist)
{
    const float eps = 1e-6f;

    CudaVec3<float> AB = p2 - p1;
    CudaVec3<float> AC = p3 - p1;
    CudaVec3<float> AQ = q - p1;

    float A00 = dot(AB, AB);
    float A11 = dot(AC, AC);
    float A01 = dot(AB, AC);
    float b0  = dot(AQ, AB);
    float b1  = dot(AQ, AC);

    float det = A00 * A11 - A01 * A01;

    float alpha = (b0 * A11 - b1 * A01) / det;
    float beta  = (b1 * A00 - b0 * A01) / det;

    if (alpha >= eps && beta >= eps && alpha + beta <= 1.0f - eps)
    {
        CudaVec3<float> p = p1 + AB * alpha + AC * beta;
        CudaVec3<float> pq = q - p;
        float n2 = dot(pq, pq);
        if (n2 >= dist2) return 0;
        float d = sqrtf(n2);
        contactP = p;
        contactQ = q;
        contactNormal = (d > 1e-15f) ? pq * (1.0f / d) : triN;
        contactDist = d;
        return 1;
    }

    float pAB = b0 / A00;
    float pAC = b1 / A11;

    enum {
        FLAG_P1  = 1<<0,
        FLAG_P2  = 1<<1,
        FLAG_P3  = 1<<2,
        FLAG_E23 = 1<<3,
        FLAG_E31 = 1<<4,
        FLAG_E12 = 1<<5,
    };

    if (pAB < eps && pAC < eps)
    {
        if (!(flags & FLAG_P1)) return 0;
        CudaVec3<float> pq = q - p1;
        float n2 = dot(pq, pq);
        if (n2 >= dist2) return 0;
        float d = sqrtf(n2);
        contactP = p1;
        contactQ = q;
        contactNormal = (d > 1e-15f) ? pq * (1.0f / d) : triN;
        contactDist = d;
        return 1;
    }
    else if (pAB < 1.0f - eps && pAB >= eps && beta < eps)
    {
        if (!(flags & FLAG_E12)) return 0;
        alpha = pAB;
        beta = 0.0f;
    }
    else if (pAC < 1.0f - eps && pAC >= eps && alpha < eps)
    {
        if (!(flags & FLAG_E31)) return 0;
        alpha = 0.0f;
        beta = pAC;
    }
    else
    {
        float pBC = (b1 - b0 + A00 - A01) / (A00 + A11 - 2.0f * A01);
        if (pBC < eps)
        {
            if (!(flags & FLAG_P2)) return 0;
            CudaVec3<float> pq = q - p2;
            float n2 = dot(pq, pq);
            if (n2 >= dist2) return 0;
            float d = sqrtf(n2);
            contactP = p2;
            contactQ = q;
            contactNormal = (d > 1e-15f) ? pq * (1.0f / d) : triN;
            contactDist = d;
            return 1;
        }
        else if (pBC > 1.0f - eps)
        {
            if (!(flags & FLAG_P3)) return 0;
            CudaVec3<float> pq = q - p3;
            float n2 = dot(pq, pq);
            if (n2 >= dist2) return 0;
            float d = sqrtf(n2);
            contactP = p3;
            contactQ = q;
            contactNormal = (d > 1e-15f) ? pq * (1.0f / d) : triN;
            contactDist = d;
            return 1;
        }
        else
        {
            if (!(flags & FLAG_E23)) return 0;
            alpha = 1.0f - pBC;
            beta  = pBC;
        }
    }

    CudaVec3<float> p = p1 + AB * alpha + AC * beta;
    CudaVec3<float> pq = q - p;
    float n2 = dot(pq, pq);
    if (n2 >= dist2) return 0;
    float d = sqrtf(n2);
    contactP = p;
    contactQ = q;
    contactNormal = (d > 1e-15f) ? pq * (1.0f / d) : triN;
    contactDist = d;
    return 1;
}

__device__ int doIntersectionLinePoint_device(
    float dist2,
    CudaVec3<float> p1, CudaVec3<float> p2,
    CudaVec3<float> q,
    CudaVec3<float>& contactP,
    CudaVec3<float>& contactQ,
    CudaVec3<float>& contactNormal,
    float& contactDist)
{
    CudaVec3<float> AB = p2 - p1;
    CudaVec3<float> AQ = q - p1;
    float A = dot(AB, AB);
    float b = dot(AQ, AB);

    float alpha = b / A;
    if (alpha < 1e-6f) alpha = 0.0f;
    else if (alpha > 1.0f - 1e-6f) alpha = 1.0f;

    CudaVec3<float> p = p1 + AB * alpha;
    CudaVec3<float> pq = q - p;
    float n2 = dot(pq, pq);
    if (n2 >= dist2) return 0;

    float d = sqrtf(n2);
    contactP = p;
    contactQ = q;
    contactNormal = (d > 1e-15f) ? pq * (1.0f / d) : CudaVec3<float>::make(1, 0, 0);
    contactDist = d;
    return 1;
}

__device__ int doIntersectionLineLine_device(
    float dist2,
    CudaVec3<float> p1, CudaVec3<float> p2,
    CudaVec3<float> q1, CudaVec3<float> q2,
    CudaVec3<float>& contactP,
    CudaVec3<float>& contactQ,
    CudaVec3<float>& contactNormal,
    float& contactDist)
{
    CudaVec3<float> AB = p2 - p1;
    CudaVec3<float> CD = q2 - q1;
    CudaVec3<float> AC = q1 - p1;

    float A00 = dot(AB, AB);
    float A11 = dot(CD, CD);
    float A01 = -dot(CD, AB);
    float b0  = dot(AB, AC);
    float b1  = -dot(CD, AC);

    float det = A00 * A11 - A01 * A01;

    float alpha, beta;
    if (fabsf(det) > 1e-12f)
    {
        alpha = (b0 * A11 - b1 * A01) / det;
        beta  = (b1 * A00 - b0 * A01) / det;
        if (alpha < 1e-6f || alpha > 1.0f - 1e-6f ||
            beta  < 1e-6f || beta  > 1.0f - 1e-6f)
            return 0;
    }
    else
    {
        alpha = 0.5f;
        beta  = 0.5f;
    }

    CudaVec3<float> p = p1 + AB * alpha;
    CudaVec3<float> q = q1 + CD * beta;
    CudaVec3<float> pq = q - p;
    float n2 = dot(pq, pq);
    if (n2 >= dist2) return 0;

    float d = sqrtf(n2);
    contactP = p;
    contactQ = q;
    contactNormal = (d > 1e-15f) ? pq * (1.0f / d) : CudaVec3<float>::make(1, 0, 0);
    contactDist = d;
    return 1;
}

__device__ int doIntersectionPointPoint_device(
    float dist2,
    CudaVec3<float> p, CudaVec3<float> q,
    CudaVec3<float>& contactP,
    CudaVec3<float>& contactQ,
    CudaVec3<float>& contactNormal,
    float& contactDist)
{
    CudaVec3<float> pq = q - p;
    float n2 = dot(pq, pq);
    if (n2 >= dist2) return 0;
    float d = sqrtf(n2);
    contactP = p;
    contactQ = q;
    contactNormal = (d > 1e-15f) ? pq * (1.0f / d) : CudaVec3<float>::make(1, 0, 0);
    contactDist = d;
    return 1;
}


__global__ void CudaNarrowPhaseDetection_doTests_kernel(
    unsigned int nbTests,
    const NarrowPhaseTestEntry* tests,
    const CudaVec3<float>* positions1,
    const CudaVec3<float>* positions2,
    const int* triangles1,
    const int* edges1,
    const int* triangles2,
    const int* edges2,
    float alarmDist2,
    NarrowPhaseResult* results)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nbTests) return;

    NarrowPhaseTestEntry test = tests[tid];
    NarrowPhaseResult res;
    res.valid = 0;
    res.elem1 = test.elem1;
    res.elem2 = test.elem2;

    CudaVec3<float> cP, cQ, cN;
    float cD;

    switch (test.type)
    {
    case TEST_TRIANGLE_POINT:
    {
        int triIdx = test.elem1;
        int ptIdx  = test.elem2;
        int v0 = triangles1[triIdx * 3 + 0];
        int v1 = triangles1[triIdx * 3 + 1];
        int v2 = triangles1[triIdx * 3 + 2];
        CudaVec3<float> p1 = positions1[v0];
        CudaVec3<float> p2 = positions1[v1];
        CudaVec3<float> p3 = positions1[v2];
        CudaVec3<float> n  = computeTriNormal(p1, p2, p3);
        CudaVec3<float> q  = positions2[ptIdx];
        res.valid = doIntersectionTrianglePoint_device(alarmDist2, test.flags, p1, p2, p3, n, q, cP, cQ, cN, cD);
        break;
    }
    case TEST_TRIANGLE_TRIVERTEX:
    {
        int triIdx1 = test.elem1;
        int triIdx2 = test.elem2;
        int vertexInTri = test.flags & 0x3;
        int triFlags = test.flags >> 2;
        int v0 = triangles1[triIdx1 * 3 + 0];
        int v1 = triangles1[triIdx1 * 3 + 1];
        int v2 = triangles1[triIdx1 * 3 + 2];
        int ptIdx = triangles2[triIdx2 * 3 + vertexInTri];
        CudaVec3<float> p1 = positions1[v0];
        CudaVec3<float> p2 = positions1[v1];
        CudaVec3<float> p3 = positions1[v2];
        CudaVec3<float> n  = computeTriNormal(p1, p2, p3);
        CudaVec3<float> q  = positions2[ptIdx];
        res.valid = doIntersectionTrianglePoint_device(alarmDist2, triFlags, p1, p2, p3, n, q, cP, cQ, cN, cD);
        break;
    }
    case TEST_TRIANGLE_EDGEVERTEX:
    {
        int triIdx  = test.elem1;
        int edgeIdx = test.elem2;
        int vertexInEdge = test.flags & 0x1;
        int triFlags = test.flags >> 1;
        int v0 = triangles1[triIdx * 3 + 0];
        int v1 = triangles1[triIdx * 3 + 1];
        int v2 = triangles1[triIdx * 3 + 2];
        int ptIdx = edges2[edgeIdx * 2 + vertexInEdge];
        CudaVec3<float> p1 = positions1[v0];
        CudaVec3<float> p2 = positions1[v1];
        CudaVec3<float> p3 = positions1[v2];
        CudaVec3<float> n  = computeTriNormal(p1, p2, p3);
        CudaVec3<float> q  = positions2[ptIdx];
        res.valid = doIntersectionTrianglePoint_device(alarmDist2, triFlags, p1, p2, p3, n, q, cP, cQ, cN, cD);
        break;
    }
    case TEST_TRIANGLE_TRIVERTEX_SWAP:
    {
        int triIdx2 = test.elem1;
        int triIdx1 = test.elem2;
        int vertexInTri = test.flags & 0x3;
        int triFlags = test.flags >> 2;
        int v0 = triangles2[triIdx2 * 3 + 0];
        int v1 = triangles2[triIdx2 * 3 + 1];
        int v2 = triangles2[triIdx2 * 3 + 2];
        int ptIdx = triangles1[triIdx1 * 3 + vertexInTri];
        CudaVec3<float> p1 = positions2[v0];
        CudaVec3<float> p2 = positions2[v1];
        CudaVec3<float> p3 = positions2[v2];
        CudaVec3<float> n  = computeTriNormal(p1, p2, p3);
        CudaVec3<float> q  = positions1[ptIdx];
        res.valid = doIntersectionTrianglePoint_device(alarmDist2, triFlags, p1, p2, p3, n, q, cP, cQ, cN, cD);
        break;
    }
    case TEST_TRIANGLE_EDGEVERTEX_SWAP:
    {
        int triIdx  = test.elem1;
        int edgeIdx = test.elem2;
        int vertexInEdge = test.flags & 0x1;
        int triFlags = test.flags >> 1;
        int v0 = triangles2[triIdx * 3 + 0];
        int v1 = triangles2[triIdx * 3 + 1];
        int v2 = triangles2[triIdx * 3 + 2];
        int ptIdx = edges1[edgeIdx * 2 + vertexInEdge];
        CudaVec3<float> p1 = positions2[v0];
        CudaVec3<float> p2 = positions2[v1];
        CudaVec3<float> p3 = positions2[v2];
        CudaVec3<float> n  = computeTriNormal(p1, p2, p3);
        CudaVec3<float> q  = positions1[ptIdx];
        res.valid = doIntersectionTrianglePoint_device(alarmDist2, triFlags, p1, p2, p3, n, q, cP, cQ, cN, cD);
        break;
    }
    case TEST_LINE_POINT:
    {
        int edgeIdx = test.elem1;
        int ptIdx   = test.elem2;
        int v0 = edges1[edgeIdx * 2 + 0];
        int v1 = edges1[edgeIdx * 2 + 1];
        CudaVec3<float> p1 = positions1[v0];
        CudaVec3<float> p2 = positions1[v1];
        CudaVec3<float> q  = positions2[ptIdx];
        res.valid = doIntersectionLinePoint_device(alarmDist2, p1, p2, q, cP, cQ, cN, cD);
        break;
    }
    case TEST_LINE_LINE:
    {
        int edgeIdx1 = test.elem1;
        int edgeIdx2 = test.elem2;
        int v0 = edges1[edgeIdx1 * 2 + 0];
        int v1 = edges1[edgeIdx1 * 2 + 1];
        int v2 = edges2[edgeIdx2 * 2 + 0];
        int v3 = edges2[edgeIdx2 * 2 + 1];
        CudaVec3<float> p1 = positions1[v0];
        CudaVec3<float> p2 = positions1[v1];
        CudaVec3<float> q1 = positions2[v2];
        CudaVec3<float> q2 = positions2[v3];
        res.valid = doIntersectionLineLine_device(alarmDist2, p1, p2, q1, q2, cP, cQ, cN, cD);
        break;
    }
    case TEST_POINT_POINT:
    {
        int ptIdx1 = test.elem1;
        int ptIdx2 = test.elem2;
        CudaVec3<float> p = positions1[ptIdx1];
        CudaVec3<float> q = positions2[ptIdx2];
        res.valid = doIntersectionPointPoint_device(alarmDist2, p, q, cP, cQ, cN, cD);
        break;
    }
    }

    if (res.valid)
    {
        res.point0   = cP;
        res.point1   = cQ;
        res.normal   = cN;
        res.distance = cD;
    }

    results[tid] = res;
}


//////////////////////
// CPU-side methods //
//////////////////////

void CudaNarrowPhaseDetection_doTests(
    unsigned int nbTests,
    const void* tests,
    const void* positions1,
    const void* positions2,
    const void* triangles1,
    const void* edges1,
    const void* triangles2,
    const void* edges2,
    float alarmDist2,
    void* results)
{
    if (nbTests == 0) return;
    dim3 threads(NARROW_BSIZE, 1);
    dim3 grid((nbTests + NARROW_BSIZE - 1) / NARROW_BSIZE, 1);
    CudaNarrowPhaseDetection_doTests_kernel<<<grid, threads>>>(
        nbTests,
        (const NarrowPhaseTestEntry*)tests,
        (const CudaVec3<float>*)positions1,
        (const CudaVec3<float>*)positions2,
        (const int*)triangles1,
        (const int*)edges1,
        (const int*)triangles2,
        (const int*)edges2,
        alarmDist2,
        (NarrowPhaseResult*)results);
    mycudaDebugError("CudaNarrowPhaseDetection_doTests_kernel");
}
