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
#include <sofa/component/collision/detection/algorithm/BVHNarrowPhase.h>
#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa::gpu::cuda
{

struct NarrowPhasePairData
{
    const void* positions1;
    const void* positions2;
    const void* triangles1;
    const void* edges1;
    const void* normals1;
    const void* triangles2;
    const void* edges2;
    const void* normals2;
    float alarmDist2;
};

struct NarrowPhaseTestEntry
{
    int type;
    int elem1;
    int elem2;
    int flags;
    int pairIndex;
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
    sofa::type::Vec3f point0;
    sofa::type::Vec3f point1;
    sofa::type::Vec3f normal;
    float distance;
    int elem1;
    int elem2;
    int valid;
    int _pad;
};

class SOFACUDA_COMPONENT_API CudaNarrowPhaseDetection
    : public sofa::component::collision::detection::algorithm::BVHNarrowPhase
{
public:
    SOFA_CLASS(CudaNarrowPhaseDetection,
               sofa::component::collision::detection::algorithm::BVHNarrowPhase);

    Data<unsigned int> d_gpuTestThreshold;

    CudaNarrowPhaseDetection();
    ~CudaNarrowPhaseDetection() override = default;

    void addCollisionPair(
        const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) override;

    void addCollisionPairs(
        const sofa::type::vector<std::pair<core::CollisionModel*, core::CollisionModel*>>& v) override;

protected:
    bool isMeshBasedPair(core::CollisionModel* cm1, core::CollisionModel* cm2) const;

    void collectLeafTests(
        core::CollisionModel* cm1, core::CollisionModel* cm2,
        core::CollisionModel* finest1, core::CollisionModel* finest2,
        int kind1, int kind2,
        bool selfCollision,
        std::vector<NarrowPhaseTestEntry>& testList);

    CudaVector<NarrowPhaseTestEntry> d_tests;
    CudaVector<NarrowPhaseResult>    d_results;
    CudaVector<NarrowPhasePairData>  d_pairData;
};

} // namespace sofa::gpu::cuda
