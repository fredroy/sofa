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
#include <sofa/component/mapping/linear/BarycentricMapping.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTriangleSetTopology.h>
using sofa::component::mapping::linear::BarycentricMapperTriangleSetTopology;
using sofa::component::mapping::linear::BarycentricMapping;

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/core/topology/BaseMeshTopology.h>
using sofa::component::topology::container::dynamic::TriangleSetTopologyContainer;
using sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer;
using sofa::core::topology::BaseMeshTopology;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;
#include <sofa/testing/BaseSimulationTest.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <random>


using sofa::type::Vec3;
using sofa::type::Vec3u;
using sofa::core::objectmodel::New;
#include <sofa/simulation/Simulation.h>
using sofa::simulation::Node ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;

#include <sofa/component/statecontainer/MechanicalObject.h>
using sofa::component::statecontainer::MechanicalObject ;

#include <sofa/simulation/Node.h>

using sofa::defaulttype::Vec3Types;

template <class In, class Out>
struct BarycentricMapperTriangleSetTopologyTest :  public BaseTest, public BarycentricMapperTriangleSetTopology<In,Out>
{
    typedef BarycentricMapperTriangleSetTopology<In,Out> Inherit;
    typedef typename In::Real Real;

    using Inherit::m_hashTable;
    using Inherit::m_hashTableSize;
    using Inherit::m_gridCellSize;
    using Inherit::m_convFactor;
    using Inherit::m_fromTopology;
    using Inherit::d_map;

    using Inherit::computeHashTable;
    using Inherit::getGridIndices;
    using Inherit::initHashing;
    using Inherit::init;

    typename In::VecCoord m_in;
    typename Out::VecCoord m_out;
    TriangleSetTopologyContainer::SPtr m_topology;

    void doSetUp() override
    {
        m_in.push_back(Vec3(0.5, 1.5, 0.0));
        m_in.push_back(Vec3(1.5, 0.0, 2.5));
        m_in.push_back(Vec3(-0.5, -1.5, 0.0));

        m_out.push_back(Vec3{-0.5, -1.5, 0.0});
        m_out.push_back(Vec3{0.5, 0.0, -10.0});

        m_topology = New<TriangleSetTopologyContainer>();
        m_fromTopology = m_topology.get();
        m_fromTopology->addTriangle(0, 1, 2);

        initHashing(m_in);
    }

    void scene_test(){
        sofa::simulation::Simulation* simu = sofa::simulation::MainSimulation::getSimulation();

        typename BarycentricMapping<In,Out>::SPtr thisObject = New<BarycentricMapping<In,Out>>();
        thisObject->setName("barycentricMapping");
        EXPECT_TRUE(thisObject->getName() == "barycentricMapping");

        const Node::SPtr node = simu->createNewGraph("root");
        const Node::SPtr nodeMapping = node->createChild("nodeToMap");
        const TriangleSetTopologyContainer::SPtr triangleContainer = New<TriangleSetTopologyContainer>();
        const TetrahedronSetTopologyContainer::SPtr tetraContainer = New<TetrahedronSetTopologyContainer>();
        const MechanicalObject<Vec3Types>::SPtr mechanical = New<MechanicalObject<Vec3Types>>();

        node->addObject(tetraContainer);
        node->addObject(mechanical);
        node->addChild(nodeMapping);
        nodeMapping->addObject(triangleContainer);
        nodeMapping->addObject(thisObject);

        EXPECT_NO_THROW(
            sofa::simulation::node::initRoot(node.get())
        );
    }

    void init_test()
    {
        init(m_out,m_in);
        EXPECT_EQ(d_map.getValue().size(),2);
    }

    void initHashing_test()
    {
        Real min =(m_in[0]-m_in[1]).norm();
        Real max =(m_in[2]-m_in[1]).norm();
        EXPECT_LE(m_gridCellSize,max);
        EXPECT_GE(m_gridCellSize,min);

        EXPECT_EQ(m_convFactor,1./m_gridCellSize);
    }
};


typedef BarycentricMapperTriangleSetTopologyTest< Vec3Types, Vec3Types> BarycentricMapperTriangleSetTopologyTest_d;


TEST_F(BarycentricMapperTriangleSetTopologyTest_d, init)
{
    EXPECT_NO_THROW(init_test());
}

TEST_F(BarycentricMapperTriangleSetTopologyTest_d, initHashing)
{
    initHashing_test();
}

/// sparseApplyJT: applyJT skips the zero entries of the mapped vector (e.g. contact forces on a few vertices of a mapped
/// surface), with exactly the same result, for the mappers of a regular grid, of a static mesh and of a dynamic topology
struct BarycentricMappingSparseApplyJT_test : public sofa::testing::BaseSimulationTest
{
    using VecDeriv = Vec3Types::VecDeriv;
    using Mapping = BarycentricMapping<Vec3Types, Vec3Types>;

    void doSetUp() override
    {
        loadPlugins({Sofa.Component.StateContainer, Sofa.Component.Mapping.Linear, Sofa.Component.Topology.Container.Grid,
                     Sofa.Component.Topology.Container.Constant, Sofa.Component.Topology.Container.Dynamic});
    }

    /// Checks applyJT, with and without sparseApplyJT, of a mapping from the topology described by its type and Data to
    /// points inside the [0, 1]^3 cube
    void checkSparseApplyJT(const std::string& topologyType, const std::map<std::string, std::string>& topologyData)
    {
        const auto root = sofa::simpleapi::createRootNode(sofa::simpleapi::createSimulation(), "root");
        sofa::simpleapi::createObject(root, "DefaultAnimationLoop");
        const auto body = sofa::simpleapi::createChild(root, "body");
        sofa::simpleapi::createObject(body, topologyType, topologyData);
        sofa::simpleapi::createObject(body, "MechanicalObject", {{"name", "dofs"}, {"template", "Vec3"}, {"position", "@topology.position"}});
        const auto points = sofa::simpleapi::createChild(body, "points");
        std::string positions;
        std::mt19937 gen(3);
        std::uniform_real_distribution<SReal> u(0.05, 0.95);
        constexpr int nbPoints = 40;
        for (int i = 0; i < nbPoints; ++i)
            positions += std::to_string(u(gen)) + " " + std::to_string(u(gen)) + " " + std::to_string(u(gen)) + " ";
        sofa::simpleapi::createObject(points, "MechanicalObject", {{"name", "dofs"}, {"template", "Vec3"}, {"position", positions}});
        auto* mapping = dynamic_cast<Mapping*>(sofa::simpleapi::createObject(points, "BarycentricMapping", {{"input", "@../dofs"}, {"output", "@dofs"}}).get());
        ASSERT_NE(mapping, nullptr);
        sofa::simulation::node::initRoot(root.get());

        BaseData* sparse = mapping->findData("sparseApplyJT");
        ASSERT_NE(sparse, nullptr) << topologyType;
        EXPECT_EQ(sparse->getValueString(), "0") << topologyType; // off by default

        auto* input = dynamic_cast<MechanicalObject<Vec3Types>*>(body->getObject("dofs"));
        ASSERT_NE(input, nullptr);
        const std::size_t nbNodes = input->getSize();

        // forces on a few points, some of them tiny (still not zero), and an accumulated force on the nodes
        sofa::Data<VecDeriv> forces, dense, skipping;
        VecDeriv f(nbPoints, Vec3(0, 0, 0));
        f[3] = Vec3(1.5, -2, 0.25);
        f[17] = Vec3(-0.5, 0.75, 3);
        f[38] = Vec3(0, 0, -1);
        f[21] = Vec3(1e-9, -2e-9, 0);
        f[30] = Vec3(0, 1e-300, 0);
        forces.setValue(f);
        VecDeriv accumulated(nbNodes);
        for (std::size_t k = 0; k < nbNodes; ++k)
            accumulated[k] = Vec3(0.1 * k, -0.2, 0.3);
        dense.setValue(accumulated);
        skipping.setValue(accumulated);

        // exact comparisons (Vec::operator== compares with a tolerance)
        const auto exactlyEqual = [](const VecDeriv& a, const VecDeriv& b)
        {
            return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
                [](const Vec3& u, const Vec3& v) { return u[0] == v[0] && u[1] == v[1] && u[2] == v[2]; });
        };

        const auto* mparams = sofa::core::mechanicalparams::defaultInstance();
        sparse->read("false");
        mapping->applyJT(mparams, dense, forces);
        sparse->read("true");
        mapping->applyJT(mparams, skipping, forces);
        EXPECT_FALSE(exactlyEqual(dense.getValue(), accumulated)) << topologyType;
        EXPECT_TRUE(exactlyEqual(dense.getValue(), skipping.getValue())) << topologyType;

        // no force at all: nothing changes
        forces.setValue(VecDeriv(nbPoints, Vec3(0, 0, 0)));
        mapping->applyJT(mparams, skipping, forces);
        EXPECT_TRUE(exactlyEqual(dense.getValue(), skipping.getValue())) << topologyType;
    }
};

TEST_F(BarycentricMappingSparseApplyJT_test, regularGrid)
{
    checkSparseApplyJT("RegularGridTopology", {{"name", "topology"}, {"n", "3 4 5"}, {"min", "0 0 0"}, {"max", "1 1 1"}});
}

/// The unit cube split into 6 positively oriented tetrahedra along its diagonal (0, 7)
const std::map<std::string, std::string> cubeTetrahedra {{"name", "topology"},
    {"position", "0 0 0  1 0 0  0 1 0  1 1 0  0 0 1  1 0 1  0 1 1  1 1 1"},
    {"tetrahedra", "0 1 3 7  0 5 1 7  0 3 2 7  0 2 6 7  0 4 5 7  0 6 4 7"}};

TEST_F(BarycentricMappingSparseApplyJT_test, meshTopology)
{
    checkSparseApplyJT("MeshTopology", cubeTetrahedra);
}

TEST_F(BarycentricMappingSparseApplyJT_test, tetrahedronSetTopology)
{
    checkSparseApplyJT("TetrahedronSetTopologyContainer", cubeTetrahedra);
}

