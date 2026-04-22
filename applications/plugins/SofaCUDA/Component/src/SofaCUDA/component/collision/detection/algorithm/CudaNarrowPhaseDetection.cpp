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

// Returns device pointer to positions. For CudaVec3Types, zero-copy from MechanicalState.
// For Vec3Types, copies into tmpBuf and returns device pointer.
template <typename DataTypes>
const void* tryGetPositionsDevice(core::CollisionModel* cm, CudaVector<float>& tmpBuf)
{
    auto* triModel = dynamic_cast<TriangleCollisionModel<DataTypes>*>(cm);
    auto* lineModel = dynamic_cast<LineCollisionModel<DataTypes>*>(cm);
    auto* pointModel = dynamic_cast<PointCollisionModel<DataTypes>*>(cm);

    core::behavior::MechanicalState<DataTypes>* mstate = nullptr;
    if (triModel) mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(triModel->getContext()->getMechanicalState());
    else if (lineModel) mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(lineModel->getContext()->getMechanicalState());
    else if (pointModel) mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(pointModel->getContext()->getMechanicalState());
    if (!mstate) return nullptr;

    const auto& pos = mstate->read(core::vec_id::read_access::position)->getValue();

    if constexpr (std::is_same_v<DataTypes, CudaVec3Types>)
    {
        return pos.deviceRead();
    }
    else
    {
        unsigned int n = (unsigned int)pos.size();
        tmpBuf.resize(n * 3);
        auto* w = tmpBuf.hostWrite();
        for (unsigned int i = 0; i < n; ++i)
        {
            w[i * 3 + 0] = (float)pos[i][0];
            w[i * 3 + 1] = (float)pos[i][1];
            w[i * 3 + 2] = (float)pos[i][2];
        }
        return tmpBuf.deviceRead();
    }
}

const void* getPositionsDevice(core::CollisionModel* cm, CudaVector<float>& tmpBuf)
{
    auto* p = tryGetPositionsDevice<CudaVec3Types>(cm, tmpBuf);
    if (p) return p;
    return tryGetPositionsDevice<sofa::defaulttype::Vec3Types>(cm, tmpBuf);
}

void uploadTriangles(core::CollisionModel* cm, CudaVector<int>& buf)
{
    auto* m = dynamic_cast<TriModel*>(cm);
    if (!m) m = reinterpret_cast<TriModel*>(dynamic_cast<CudaTriModel*>(cm));
    // Both TriModel and CudaTriModel derive from TriangleCollisionModel which has getTriangles()
    const sofa::core::topology::BaseMeshTopology::SeqTriangles* tris = nullptr;
    if (auto* t = dynamic_cast<TriModel*>(cm)) tris = &t->getTriangles();
    else if (auto* t = dynamic_cast<CudaTriModel*>(cm)) tris = &t->getTriangles();
    if (!tris || tris->empty()) return;
    unsigned int n = (unsigned int)tris->size();
    buf.resize(n * 3);
    auto* w = buf.hostWrite();
    for (unsigned int i = 0; i < n; ++i)
    {
        w[i * 3 + 0] = (int)(*tris)[i][0];
        w[i * 3 + 1] = (int)(*tris)[i][1];
        w[i * 3 + 2] = (int)(*tris)[i][2];
    }
}

void uploadEdges(core::CollisionModel* cm, CudaVector<int>& buf)
{
    auto* topo = cm->getCollisionTopology();
    if (!topo) return;
    const auto& edges = topo->getEdges();
    if (edges.empty()) return;
    unsigned int n = (unsigned int)edges.size();
    buf.resize(n * 2);
    auto* w = buf.hostWrite();
    for (unsigned int i = 0; i < n; ++i)
    {
        w[i * 2 + 0] = (int)edges[i][0];
        w[i * 2 + 1] = (int)edges[i][1];
    }
}

struct TestInfo { int type; int elem1; int elem2; int flags; bool swapResult; };

} // anonymous namespace


bool CudaNarrowPhaseDetection::isMeshBasedPair(
    core::CollisionModel* cm1, core::CollisionModel* cm2) const
{
    return classifyModel(cm1) != MODEL_OTHER && classifyModel(cm2) != MODEL_OTHER;
}


void CudaNarrowPhaseDetection::addCollisionPair(
    const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair)
{
    core::CollisionModel* cm1 = cmPair.first;
    core::CollisionModel* cm2 = cmPair.second;

    if (!cm1->isSimulated() && !cm2->isSimulated())
        return;
    if (cm1->empty() || cm2->empty())
        return;

    if (isMeshBasedPair(cm1->getLast(), cm2->getLast()))
        collectTestsAndRunGPU(cmPair);
    else
        BVHNarrowPhase::addCollisionPair(cmPair);
}


void CudaNarrowPhaseDetection::collectTestsAndRunGPU(
    const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair)
{
    core::CollisionModel* cm1 = cmPair.first;
    core::CollisionModel* cm2 = cmPair.second;

    core::CollisionModel* finest1 = cm1->getLast();
    core::CollisionModel* finest2 = cm2->getLast();

    const bool selfCollision = isSelfCollision(finest1, finest2);

    bool swapModels = false;
    core::collision::ElementIntersector* finestIntersector =
        intersectionMethod->findIntersector(finest1, finest2, swapModels);
    if (!finestIntersector) return;

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

    const float alarmDist  = (float)(intersectionMethod->getAlarmDistance() + finest1->getContactDistance() + finest2->getContactDistance());
    const float contactDist = (float)(intersectionMethod->getContactDistance() + finest1->getContactDistance() + finest2->getContactDistance());
    const float alarmDist2 = alarmDist * alarmDist;

    // BVH traversal to collect leaf tests
    std::vector<TestInfo> testList;
    testList.reserve(4096);

    core::CollisionModel* bvhFinest1 = finest1;
    core::CollisionModel* bvhFinest2 = finest2;
    if (finest1 == cm1 || finest2 == cm2)
    {
        bvhFinest1 = nullptr;
        bvhFinest2 = nullptr;
    }

    using CollisionIteratorRange = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>;
    using TestPair = std::pair<CollisionIteratorRange, CollisionIteratorRange>;

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
                            testList.push_back({TEST_TRIANGLE_POINT, idx1, idx2, getTriangleFlags(finest1, idx1), false});
                        }
                        else if (kind1 == MODEL_POINT && kind2 == MODEL_TRIANGLE)
                        {
                            testList.push_back({TEST_TRIANGLE_POINT, idx2, idx1, getTriangleFlags(finest2, idx2), true});
                        }
                        else if (kind1 == MODEL_LINE && kind2 == MODEL_POINT)
                        {
                            testList.push_back({TEST_LINE_POINT, idx1, idx2, 0, false});
                        }
                        else if (kind1 == MODEL_POINT && kind2 == MODEL_LINE)
                        {
                            testList.push_back({TEST_LINE_POINT, idx2, idx1, 0, true});
                        }
                        else if (kind1 == MODEL_LINE && kind2 == MODEL_LINE)
                        {
                            testList.push_back({TEST_LINE_LINE, idx1, idx2, 0, false});
                        }
                        else if (kind1 == MODEL_POINT && kind2 == MODEL_POINT)
                        {
                            testList.push_back({TEST_POINT_POINT, idx1, idx2, 0, false});
                        }
                        else if (kind1 == MODEL_TRIANGLE && kind2 == MODEL_TRIANGLE)
                        {
                            int f1 = getTriangleFlags(finest1, idx1);
                            int f2 = getTriangleFlags(finest2, idx2);
                            if (f2 & (1<<0))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX, idx1, idx2, (f1 << 2) | 0, false});
                            if (f2 & (1<<1))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX, idx1, idx2, (f1 << 2) | 1, false});
                            if (f2 & (1<<2))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX, idx1, idx2, (f1 << 2) | 2, false});
                            if (f1 & (1<<0))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX_SWAP, idx2, idx1, (f2 << 2) | 0, true});
                            if (f1 & (1<<1))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX_SWAP, idx2, idx1, (f2 << 2) | 1, true});
                            if (f1 & (1<<2))
                                testList.push_back({TEST_TRIANGLE_TRIVERTEX_SWAP, idx2, idx1, (f2 << 2) | 2, true});
                        }
                        else if (kind1 == MODEL_TRIANGLE && kind2 == MODEL_LINE)
                        {
                            int f1 = getTriangleFlags(finest1, idx1);
                            testList.push_back({TEST_TRIANGLE_EDGEVERTEX, idx1, idx2, (f1 << 1) | 0, false});
                            testList.push_back({TEST_TRIANGLE_EDGEVERTEX, idx1, idx2, (f1 << 1) | 1, false});
                        }
                        else if (kind1 == MODEL_LINE && kind2 == MODEL_TRIANGLE)
                        {
                            int f2 = getTriangleFlags(finest2, idx2);
                            testList.push_back({TEST_TRIANGLE_EDGEVERTEX_SWAP, idx2, idx1, (f2 << 1) | 0, true});
                            testList.push_back({TEST_TRIANGLE_EDGEVERTEX_SWAP, idx2, idx1, (f2 << 1) | 1, true});
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

    if (testList.empty())
        return;

    if (testList.size() < d_gpuTestThreshold.getValue())
    {
        BVHNarrowPhase::addCollisionPair(cmPair);
        return;
    }

    // Get device pointers for positions (zero-copy for CudaVec3Types)
    CudaVector<float> tmpPos1, tmpPos2;
    const void* devPos1 = getPositionsDevice(finest1, tmpPos1);
    const void* devPos2 = getPositionsDevice(finest2, tmpPos2);
    if (!devPos1 || !devPos2)
    {
        BVHNarrowPhase::addCollisionPair(cmPair);
        return;
    }

    // Upload topology (triangles/edges) — static data
    CudaVector<int> gpuTri1, gpuTri2, gpuEdge1, gpuEdge2;
    uploadTriangles(finest1, gpuTri1);
    uploadTriangles(finest2, gpuTri2);
    uploadEdges(finest1, gpuEdge1);
    uploadEdges(finest2, gpuEdge2);

    // Upload test entries
    unsigned int nbTests = (unsigned int)testList.size();
    d_tests.resize(nbTests);
    {
        auto* w = d_tests.hostWrite();
        for (unsigned int i = 0; i < nbTests; ++i)
        {
            w[i].type  = testList[i].type;
            w[i].elem1 = testList[i].elem1;
            w[i].elem2 = testList[i].elem2;
            w[i].flags = testList[i].flags;
        }
    }

    d_results.resize(nbTests);

    CudaNarrowPhaseDetection_doTests(
        nbTests,
        d_tests.deviceRead(),
        devPos1,
        devPos2,
        gpuTri1.empty() ? nullptr : gpuTri1.deviceRead(),
        gpuEdge1.empty() ? nullptr : gpuEdge1.deviceRead(),
        gpuTri2.empty() ? nullptr : gpuTri2.deviceRead(),
        gpuEdge2.empty() ? nullptr : gpuEdge2.deviceRead(),
        alarmDist2,
        d_results.deviceWrite());

    // Read results and fill outputs
    const auto* results = d_results.hostRead();
    auto* typedOutputs = dynamic_cast<sofa::type::vector<sofa::core::collision::DetectionOutput>*>(outputs);
    if (!typedOutputs)
        return;

    for (unsigned int i = 0; i < nbTests; ++i)
    {
        if (!results[i].valid) continue;

        const auto& r = results[i];
        const bool isSwap = testList[i].swapResult;

        sofa::core::collision::DetectionOutput det;
        det.value = (double)r.distance - (double)contactDist;

        if (isSwap)
        {
            det.point[0] = sofa::type::Vec3(r.point1[0], r.point1[1], r.point1[2]);
            det.point[1] = sofa::type::Vec3(r.point0[0], r.point0[1], r.point0[2]);
            det.normal = sofa::type::Vec3(-r.normal[0], -r.normal[1], -r.normal[2]);
            det.elem.first  = core::CollisionElementIterator(finest1, r.elem2);
            det.elem.second = core::CollisionElementIterator(finest2, r.elem1);
            det.id = r.elem1;
        }
        else
        {
            det.point[0] = sofa::type::Vec3(r.point0[0], r.point0[1], r.point0[2]);
            det.point[1] = sofa::type::Vec3(r.point1[0], r.point1[1], r.point1[2]);
            det.normal = sofa::type::Vec3(r.normal[0], r.normal[1], r.normal[2]);
            det.elem.first  = core::CollisionElementIterator(finest1, r.elem1);
            det.elem.second = core::CollisionElementIterator(finest2, r.elem2);
            det.id = r.elem2;
        }

        typedOutputs->push_back(det);
    }
}

} // namespace sofa::gpu::cuda
