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
#include <SofaCUDA/component/collision/detection/algorithm/CudaNarrowPhaseDetection.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>
#include <sofa/component/collision/geometry/LineCollisionModel.h>
#include <sofa/component/collision/geometry/PointCollisionModel.h>
#include <sofa/component/collision/detection/intersection/NewProximityIntersection.h>
#include <sofa/component/collision/detection/algorithm/MirrorIntersector.h>

#include <queue>
#include <stack>
#include <map>
#include <memory>

extern "C"
{
    void CudaNarrowPhaseDetection_doTests(
        unsigned int nbTests,
        const void* tests,
        unsigned int nbPairs,
        const void* pairData,
        void* results);
}

namespace sofa::gpu::cuda
{

using namespace sofa::component::collision::geometry;
using namespace sofa::component::collision::detection::algorithm;
using namespace sofa::component::collision::detection::intersection;
using TriModel = TriangleCollisionModel<sofa::defaulttype::Vec3Types>;
using CudaTriModel = TriangleCollisionModel<CudaVec3Types>;

void registerCudaNarrowPhaseDetection(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        sofa::core::ObjectRegistrationData(
            "GPU-accelerated narrow phase collision detection based on CUDA.")
        .add<CudaNarrowPhaseDetection>());
}

CudaNarrowPhaseDetection::CudaNarrowPhaseDetection()
    : d_gpuTestThreshold(initData(&d_gpuTestThreshold, (unsigned int)128,
          "gpuTestThreshold",
          "Minimum number of primitive tests to dispatch to GPU (below this, CPU is used)"))
{
}

namespace
{

enum ModelKind { MODEL_TRIANGLE, MODEL_LINE, MODEL_POINT, MODEL_OTHER };

template <typename DataTypes>
ModelKind getModelKind(core::CollisionModel* cm)
{
    if (dynamic_cast<TriangleCollisionModel<DataTypes>*>(cm)) return MODEL_TRIANGLE;
    if (dynamic_cast<LineCollisionModel<DataTypes>*>(cm))     return MODEL_LINE;
    if (dynamic_cast<PointCollisionModel<DataTypes>*>(cm))    return MODEL_POINT;
    return MODEL_OTHER;
}

ModelKind classifyModel(core::CollisionModel* cm)
{
    auto k = getModelKind<CudaVec3Types>(cm);
    if (k != MODEL_OTHER) return k;
    return getModelKind<sofa::defaulttype::Vec3Types>(cm);
}

int getTriangleFlags(core::CollisionModel* cm, int idx)
{
    if (auto* m = dynamic_cast<TriModel*>(cm))
        return m->getTriangleFlags(idx);
    if (auto* m = dynamic_cast<CudaTriModel*>(cm))
        return m->getTriangleFlags(idx);
    return 0x3F;
}

struct GpuMeshPtrs
{
    const void* positions = nullptr;
    const void* triangles = nullptr;
    const void* edges     = nullptr;
    const void* normals   = nullptr;

    CudaVector<float> ownedPositions;
    CudaVector<int>   ownedTriangles;
    CudaVector<int>   ownedEdges;
    CudaVector<float> ownedNormals;
};

template <typename DataTypes>
bool tryFillGpuPtrs(core::CollisionModel* cm, GpuMeshPtrs& ptrs)
{
    auto* triModel = dynamic_cast<TriangleCollisionModel<DataTypes>*>(cm);
    auto* lineModel = dynamic_cast<LineCollisionModel<DataTypes>*>(cm);
    auto* pointModel = dynamic_cast<PointCollisionModel<DataTypes>*>(cm);

    core::behavior::MechanicalState<DataTypes>* mstate = nullptr;
    if (triModel)
        mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(triModel->getContext()->getMechanicalState());
    else if (lineModel)
        mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(lineModel->getContext()->getMechanicalState());
    else if (pointModel)
        mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(pointModel->getContext()->getMechanicalState());

    if (!mstate) return false;

    const auto& posData = mstate->read(core::vec_id::read_access::position)->getValue();

    if constexpr (std::is_same_v<DataTypes, CudaVec3Types>)
    {
        ptrs.positions = posData.deviceRead();
    }
    else
    {
        unsigned int nVerts = (unsigned int)posData.size();
        ptrs.ownedPositions.resize(nVerts * 3);
        auto* w = ptrs.ownedPositions.hostWrite();
        for (unsigned int i = 0; i < nVerts; ++i)
        {
            w[i * 3 + 0] = (float)posData[i][0];
            w[i * 3 + 1] = (float)posData[i][1];
            w[i * 3 + 2] = (float)posData[i][2];
        }
        ptrs.positions = ptrs.ownedPositions.deviceRead();
    }

    if (triModel)
    {
        const auto& tris = triModel->getTriangles();
        unsigned int nTris = (unsigned int)tris.size();
        ptrs.ownedTriangles.resize(nTris * 3);
        {
            auto* w = ptrs.ownedTriangles.hostWrite();
            for (unsigned int i = 0; i < nTris; ++i)
            {
                w[i * 3 + 0] = (int)tris[i][0];
                w[i * 3 + 1] = (int)tris[i][1];
                w[i * 3 + 2] = (int)tris[i][2];
            }
        }
        ptrs.triangles = ptrs.ownedTriangles.deviceRead();

        const auto& normals = triModel->getNormals();
        unsigned int nNormals = std::min((unsigned int)normals.size(), nTris);
        ptrs.ownedNormals.resize(nTris * 3);
        {
            auto* w = ptrs.ownedNormals.hostWrite();
            for (unsigned int i = 0; i < nNormals; ++i)
            {
                w[i * 3 + 0] = (float)normals[i][0];
                w[i * 3 + 1] = (float)normals[i][1];
                w[i * 3 + 2] = (float)normals[i][2];
            }
        }
        ptrs.normals = ptrs.ownedNormals.deviceRead();
    }

    auto* topo = cm->getCollisionTopology();
    if (topo)
    {
        const auto& edges = topo->getEdges();
        unsigned int nEdges = (unsigned int)edges.size();
        if (nEdges > 0)
        {
            ptrs.ownedEdges.resize(nEdges * 2);
            {
                auto* w = ptrs.ownedEdges.hostWrite();
                for (unsigned int i = 0; i < nEdges; ++i)
                {
                    w[i * 2 + 0] = (int)edges[i][0];
                    w[i * 2 + 1] = (int)edges[i][1];
                }
            }
            ptrs.edges = ptrs.ownedEdges.deviceRead();
        }
    }
    return true;
}

bool fillGpuPtrs(core::CollisionModel* cm, GpuMeshPtrs& ptrs)
{
    if (tryFillGpuPtrs<CudaVec3Types>(cm, ptrs)) return true;
    return tryFillGpuPtrs<sofa::defaulttype::Vec3Types>(cm, ptrs);
}

struct PairContext
{
    core::CollisionModel* finest1;
    core::CollisionModel* finest2;
    sofa::core::collision::DetectionOutputVector** outputs;
    float contactDist;
    unsigned int testBegin;
    unsigned int testEnd;
};

} // anonymous namespace


bool CudaNarrowPhaseDetection::isMeshBasedPair(
    core::CollisionModel* cm1, core::CollisionModel* cm2) const
{
    return classifyModel(cm1) != MODEL_OTHER && classifyModel(cm2) != MODEL_OTHER;
}


void CudaNarrowPhaseDetection::collectLeafTests(
    core::CollisionModel* cm1, core::CollisionModel* cm2,
    core::CollisionModel* finest1, core::CollisionModel* finest2,
    int kind1, int kind2,
    bool selfCollision,
    std::vector<NarrowPhaseTestEntry>& testList)
{
    using CollisionIteratorRange = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>;
    using TestPair = std::pair<CollisionIteratorRange, CollisionIteratorRange>;

    core::CollisionModel* bvhFinest1 = finest1;
    core::CollisionModel* bvhFinest2 = finest2;
    if (finest1 == cm1 || finest2 == cm2)
    {
        bvhFinest1 = nullptr;
        bvhFinest2 = nullptr;
    }

    std::queue<TestPair> externalCells;
    initializeExternalCells(cm1, cm2, externalCells);

    core::collision::ElementIntersector* coarseIntersector = nullptr;
    MirrorIntersector mirror;
    core::CollisionModel* cachedCm1 = nullptr;
    core::CollisionModel* cachedCm2 = nullptr;

    while (!externalCells.empty())
    {
        TestPair cell = externalCells.front();
        externalCells.pop();

        auto* cellCm1 = cell.first.first.getCollisionModel();
        auto* cellCm2 = cell.second.first.getCollisionModel();

        if (cellCm1 != cachedCm1 || cellCm2 != cachedCm2)
        {
            cachedCm1 = cellCm1;
            cachedCm2 = cellCm2;
            bool swap = false;
            coarseIntersector = intersectionMethod->findIntersector(cellCm1, cellCm2, swap);
            if (swap)
            {
                mirror.intersector = intersectionMethod->findIntersector(cellCm2, cellCm1, swap);
                coarseIntersector = &mirror;
            }
        }
        if (!coarseIntersector) continue;

        std::stack<TestPair> internalCells;
        internalCells.push(cell);

        while (!internalCells.empty())
        {
            TestPair current = internalCells.top();
            internalCells.pop();

            auto* curCm1 = current.first.first.getCollisionModel();
            auto* curCm2 = current.second.first.getCollisionModel();

            bool atFinest = (bvhFinest1 ? curCm1 == bvhFinest1 : curCm1 == finest1) &&
                            (bvhFinest2 ? curCm2 == bvhFinest2 : curCm2 == finest2);

            if (atFinest)
            {
                for (auto it1 = current.first.first; it1 != current.first.second; ++it1)
                {
                    for (auto it2 = current.second.first; it2 != current.second.second; ++it2)
                    {
                        if (selfCollision && !it1.canCollideWith(it2))
                            continue;

                        int idx1 = (int)it1.getIndex();
                        int idx2 = (int)it2.getIndex();

                        if (kind1 == MODEL_TRIANGLE && kind2 == MODEL_POINT)
                        {
                            testList.push_back({TEST_TRIANGLE_POINT, idx1, idx2, getTriangleFlags(finest1, idx1), 0});
                        }
                        else if (kind1 == MODEL_POINT && kind2 == MODEL_TRIANGLE)
                        {
                            testList.push_back({TEST_TRIANGLE_POINT, idx2, idx1, getTriangleFlags(finest2, idx2), 0});
                        }
                        else if (kind1 == MODEL_LINE && kind2 == MODEL_POINT)
                        {
                            testList.push_back({TEST_LINE_POINT, idx1, idx2, 0, 0});
                        }
                        else if (kind1 == MODEL_POINT && kind2 == MODEL_LINE)
                        {
                            testList.push_back({TEST_LINE_POINT, idx2, idx1, 0, 0});
                        }
                        else if (kind1 == MODEL_LINE && kind2 == MODEL_LINE)
                        {
                            testList.push_back({TEST_LINE_LINE, idx1, idx2, 0, 0});
                        }
                        else if (kind1 == MODEL_POINT && kind2 == MODEL_POINT)
                        {
                            testList.push_back({TEST_POINT_POINT, idx1, idx2, 0, 0});
                        }
                        else if (kind1 == MODEL_TRIANGLE && kind2 == MODEL_TRIANGLE)
                        {
                            int f1 = getTriangleFlags(finest1, idx1);
                            int f2 = getTriangleFlags(finest2, idx2);
                            if (f2 & (1<<0))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX, idx1, idx2, (f1 << 2) | 0, 0});
                            if (f2 & (1<<1))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX, idx1, idx2, (f1 << 2) | 1, 0});
                            if (f2 & (1<<2))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX, idx1, idx2, (f1 << 2) | 2, 0});
                            if (f1 & (1<<0))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX_SWAP, idx2, idx1, (f2 << 2) | 0, 0});
                            if (f1 & (1<<1))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX_SWAP, idx2, idx1, (f2 << 2) | 1, 0});
                            if (f1 & (1<<2))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX_SWAP, idx2, idx1, (f2 << 2) | 2, 0});
                        }
                        else if (kind1 == MODEL_TRIANGLE && kind2 == MODEL_LINE)
                        {
                            int f1 = getTriangleFlags(finest1, idx1);
                            testList.push_back({TEST_TRIANGLE_EDGEVERTEX, idx1, idx2, (f1 << 1) | 0, 0});
                            testList.push_back({TEST_TRIANGLE_EDGEVERTEX, idx1, idx2, (f1 << 1) | 1, 0});
                        }
                        else if (kind1 == MODEL_LINE && kind2 == MODEL_TRIANGLE)
                        {
                            int f2 = getTriangleFlags(finest2, idx2);
                            testList.push_back({TEST_TRIANGLE_EDGEVERTEX_SWAP, idx2, idx1, (f2 << 1) | 0, 0});
                            testList.push_back({TEST_TRIANGLE_EDGEVERTEX_SWAP, idx2, idx1, (f2 << 1) | 1, 0});
                        }
                    }
                }
            }
            else
            {
                for (auto it1 = current.first.first; it1 != current.first.second; ++it1)
                {
                    for (auto it2 = current.second.first; it2 != current.second.second; ++it2)
                    {
                        if (!coarseIntersector->canIntersect(it1, it2, intersectionMethod))
                            continue;

                        auto intChild1 = it1.getInternalChildren();
                        auto intChild2 = it2.getInternalChildren();
                        bool hasInt1 = intChild1.first != intChild1.second;
                        bool hasInt2 = intChild2.first != intChild2.second;

                        if (hasInt1 && hasInt2)
                        {
                            internalCells.push({intChild1, intChild2});
                        }
                        else if (hasInt1)
                        {
                            auto singleton2 = CollisionIteratorRange{it2, core::CollisionElementIterator(it2.getCollisionModel(), it2.getIndex() + 1)};
                            internalCells.push({intChild1, singleton2});
                        }
                        else if (hasInt2)
                        {
                            auto singleton1 = CollisionIteratorRange{it1, core::CollisionElementIterator(it1.getCollisionModel(), it1.getIndex() + 1)};
                            internalCells.push({singleton1, intChild2});
                        }
                        else
                        {
                            auto extChild1 = it1.getExternalChildren();
                            auto extChild2 = it2.getExternalChildren();
                            bool hasExt1 = extChild1.first != extChild1.second;
                            bool hasExt2 = extChild2.first != extChild2.second;

                            if (hasExt1 && hasExt2)
                            {
                                externalCells.push({extChild1, extChild2});
                            }
                            else if (hasExt1)
                            {
                                auto singleton2 = CollisionIteratorRange{it2, core::CollisionElementIterator(it2.getCollisionModel(), it2.getIndex() + 1)};
                                externalCells.push({extChild1, singleton2});
                            }
                            else if (hasExt2)
                            {
                                auto singleton1 = CollisionIteratorRange{it1, core::CollisionElementIterator(it1.getCollisionModel(), it1.getIndex() + 1)};
                                externalCells.push({singleton1, extChild2});
                            }
                        }
                    }
                }
            }
        }
    }
}


void CudaNarrowPhaseDetection::addCollisionPair(
    const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair)
{
    sofa::type::vector<std::pair<core::CollisionModel*, core::CollisionModel*>> v;
    v.push_back(cmPair);
    addCollisionPairs(v);
}


void CudaNarrowPhaseDetection::addCollisionPairs(
    const sofa::type::vector<std::pair<core::CollisionModel*, core::CollisionModel*>>& v)
{
    std::vector<NarrowPhaseTestEntry> allTests;
    allTests.reserve(16384);

    std::vector<PairContext> gpuPairs;
    gpuPairs.reserve(v.size());

    std::map<core::CollisionModel*, std::shared_ptr<GpuMeshPtrs>> meshCache;

    for (const auto& cmPair : v)
    {
        core::CollisionModel* cm1 = cmPair.first;
        core::CollisionModel* cm2 = cmPair.second;

        if (!cm1->isSimulated() && !cm2->isSimulated())
            continue;
        if (cm1->empty() || cm2->empty())
            continue;

        if (!isMeshBasedPair(cm1->getLast(), cm2->getLast()))
        {
            BVHNarrowPhase::addCollisionPair(cmPair);
            continue;
        }

        core::CollisionModel* finest1 = cm1->getLast();
        core::CollisionModel* finest2 = cm2->getLast();

        const bool selfCollision = isSelfCollision(finest1, finest2);

        bool swapModels = false;
        core::collision::ElementIntersector* finestIntersector =
            intersectionMethod->findIntersector(finest1, finest2, swapModels);
        if (!finestIntersector) continue;

        if (swapModels)
        {
            std::swap(cm1, cm2);
            std::swap(finest1, finest2);
        }

        sofa::core::collision::DetectionOutputVector*& outputs =
            this->getDetectionOutputs(finest1, finest2);
        finestIntersector->beginIntersect(finest1, finest2, outputs);

        const ModelKind kind1 = classifyModel(finest1);
        const ModelKind kind2 = classifyModel(finest2);

        const float contactDist = (float)(intersectionMethod->getContactDistance() + finest1->getContactDistance() + finest2->getContactDistance());

        unsigned int testBegin = (unsigned int)allTests.size();
        collectLeafTests(cm1, cm2, finest1, finest2, kind1, kind2,
                         selfCollision, allTests);
        unsigned int testEnd = (unsigned int)allTests.size();

        if (testEnd == testBegin)
            continue;

        auto getOrCreate = [&](core::CollisionModel* cm) -> std::shared_ptr<GpuMeshPtrs> {
            auto it = meshCache.find(cm);
            if (it != meshCache.end()) return it->second;
            auto data = std::make_shared<GpuMeshPtrs>();
            if (!fillGpuPtrs(cm, *data))
                return nullptr;
            meshCache[cm] = data;
            return data;
        };

        auto mesh1 = getOrCreate(finest1);
        auto mesh2 = getOrCreate(finest2);
        if (!mesh1 || !mesh2)
        {
            allTests.resize(testBegin);
            BVHNarrowPhase::addCollisionPair(cmPair);
            continue;
        }

        PairContext ctx;
        ctx.finest1 = finest1;
        ctx.finest2 = finest2;
        ctx.outputs = &outputs;
        ctx.contactDist = contactDist;
        ctx.testBegin = testBegin;
        ctx.testEnd = testEnd;
        gpuPairs.push_back(ctx);
    }

    if (allTests.empty())
        return;

    if (allTests.size() < d_gpuTestThreshold.getValue())
    {
        for (const auto& ctx : gpuPairs)
            BVHNarrowPhase::addCollisionPair({ctx.finest1->getFirst(), ctx.finest2->getFirst()});
        return;
    }

    // Build per-pair GPU data array
    unsigned int nbPairs = (unsigned int)gpuPairs.size();
    d_pairData.resize(nbPairs);
    {
        auto* w = d_pairData.hostWrite();
        for (unsigned int p = 0; p < nbPairs; ++p)
        {
            auto& ctx = gpuPairs[p];
            auto& m1 = *meshCache[ctx.finest1];
            auto& m2 = *meshCache[ctx.finest2];

            auto& pd = w[p];
            pd.positions1 = m1.positions;
            pd.positions2 = m2.positions;
            pd.triangles1 = m1.triangles;
            pd.edges1     = m1.edges;
            pd.normals1   = m1.normals;
            pd.triangles2 = m2.triangles;
            pd.edges2     = m2.edges;
            pd.normals2   = m2.normals;

            const float alarmDist = (float)(intersectionMethod->getAlarmDistance() + ctx.finest1->getContactDistance() + ctx.finest2->getContactDistance());
            pd.alarmDist2 = alarmDist * alarmDist;
        }
    }

    // Upload all tests with pairIndex
    unsigned int nbTests = (unsigned int)allTests.size();
    d_tests.resize(nbTests);
    {
        auto* w = d_tests.hostWrite();
        for (unsigned int p = 0; p < nbPairs; ++p)
        {
            for (unsigned int t = gpuPairs[p].testBegin; t < gpuPairs[p].testEnd; ++t)
            {
                w[t] = allTests[t];
                w[t].pairIndex = (int)p;
            }
        }
    }

    d_results.resize(nbTests);

    CudaNarrowPhaseDetection_doTests(
        nbTests,
        d_tests.deviceRead(),
        nbPairs,
        d_pairData.deviceRead(),
        d_results.deviceWrite());

    const auto* results = d_results.hostRead();

    for (unsigned int p = 0; p < nbPairs; ++p)
    {
        auto& ctx = gpuPairs[p];
        auto* typedOutputs = dynamic_cast<sofa::type::vector<sofa::core::collision::DetectionOutput>*>(*ctx.outputs);
        if (!typedOutputs)
            continue;

        for (unsigned int i = ctx.testBegin; i < ctx.testEnd; ++i)
        {
            if (!results[i].valid) continue;

            const auto& r = results[i];
            const int testType = allTests[i].type;
            const bool isSwap = (testType == TEST_TRIANGLE_TRIVERTEX_SWAP ||
                                 testType == TEST_TRIANGLE_EDGEVERTEX_SWAP);

            sofa::core::collision::DetectionOutput det;
            det.value = (double)r.distance - (double)ctx.contactDist;

            if (isSwap)
            {
                det.point[0] = sofa::type::Vec3(r.point1[0], r.point1[1], r.point1[2]);
                det.point[1] = sofa::type::Vec3(r.point0[0], r.point0[1], r.point0[2]);
                det.normal = sofa::type::Vec3(-r.normal[0], -r.normal[1], -r.normal[2]);
                det.elem.first  = core::CollisionElementIterator(ctx.finest1, r.elem2);
                det.elem.second = core::CollisionElementIterator(ctx.finest2, r.elem1);
                det.id = r.elem1;
            }
            else
            {
                det.point[0] = sofa::type::Vec3(r.point0[0], r.point0[1], r.point0[2]);
                det.point[1] = sofa::type::Vec3(r.point1[0], r.point1[1], r.point1[2]);
                det.normal = sofa::type::Vec3(r.normal[0], r.normal[1], r.normal[2]);
                det.elem.first  = core::CollisionElementIterator(ctx.finest1, r.elem1);
                det.elem.second = core::CollisionElementIterator(ctx.finest2, r.elem2);
                det.id = r.elem2;
            }

            typedOutputs->push_back(det);
        }
    }
}

} // namespace sofa::gpu::cuda
