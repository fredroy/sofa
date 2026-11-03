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
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/mapping/linear/IdentityMapping.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace
{

using defaulttype::Vec3Types;
using component::statecontainer::MechanicalObject;

/// Identity mapping counting its applications to the velocities/displacements (applyJ) and to the forces (applyJT)
class CountingIdentityMapping : public component::mapping::linear::IdentityMapping<Vec3Types, Vec3Types>
{
public:
    SOFA_CLASS(CountingIdentityMapping, SOFA_TEMPLATE2(component::mapping::linear::IdentityMapping, Vec3Types, Vec3Types));

    void applyJ(const core::MechanicalParams* mparams, Data<VecDeriv>& out, const Data<InVecDeriv>& in) override
    {
        ++nbApplyJ;
        Inherit1::applyJ(mparams, out, in);
    }

    void applyJT(const core::MechanicalParams* mparams, Data<InVecDeriv>& out, const Data<VecDeriv>& in) override
    {
        ++nbApplyJT;
        Inherit1::applyJT(mparams, out, in);
    }

    unsigned int nbApplyJ { 0 };
    unsigned int nbApplyJT { 0 };
};

/// Three particles falling under gravity, held by springs to their rest positions which act on a mapped copy of the
/// particles (identity mapping), with Rayleigh damping. The mass is on the particles, or on their mapped copy.
struct MappedStatesScene
{
    simulation::Node::SPtr root;
    CountingIdentityMapping::SPtr mapping;
    MechanicalObject<Vec3Types>* particles { nullptr };
    MechanicalObject<Vec3Types>* mapped { nullptr };

    explicit MappedStatesScene(bool massOnMappedState)
    {
        root = simpleapi::createRootNode(simpleapi::createSimulation(), "root");
        root->setGravity({0, -9.81, 0});
        simpleapi::createObject(root, "DefaultAnimationLoop");

        const auto body = simpleapi::createChild(root, "body");
        simpleapi::createObject(body, "EulerImplicitIntegrationScheme", {{"rayleighStiffness", "0.1"}, {"rayleighMass", "0.1"}});
        simpleapi::createObject(body, "SparseLDLSolver", {{"template", "CompressedRowSparseMatrixd"}});
        const std::string positions = "0 0 0  1 0 0  2 0.5 0";
        particles = dynamic_cast<MechanicalObject<Vec3Types>*>(
            simpleapi::createObject(body, "MechanicalObject", {{"template", "Vec3"}, {"position", positions}}).get());
        if (!massOnMappedState)
            simpleapi::createObject(body, "UniformMass", {{"totalMass", "3"}});

        const auto copy = simpleapi::createChild(body, "mappedCopy");
        mapped = dynamic_cast<MechanicalObject<Vec3Types>*>(
            simpleapi::createObject(copy, "MechanicalObject", {{"template", "Vec3"}, {"position", positions}}).get());
        mapping = core::objectmodel::New<CountingIdentityMapping>();
        mapping->setModels(particles, mapped);
        copy->addObject(mapping);
        simpleapi::createObject(copy, "RestShapeSpringsForceField", {{"stiffness", "200"}});
        if (massOnMappedState)
            simpleapi::createObject(copy, "UniformMass", {{"totalMass", "3"}});

        sofa::simulation::node::initRoot(root.get());
    }

    void step(int nbSteps)
    {
        for (int i = 0; i < nbSteps; ++i)
            sofa::simulation::node::animate(root.get(), 0.01);
    }

    const type::vector<type::Vec3>& positions(const MechanicalObject<Vec3Types>* state) const
    {
        return state->read(core::vec_id::read_access::position)->getValue();
    }
};

struct EulerImplicitIntegrationSchemeMappedStates_test : public BaseSimulationTest
{
    void doSetUp() override
    {
        loadPlugins({Sofa.Component.IntegrationScheme.Backward, Sofa.Component.LinearSolver.Direct, Sofa.Component.StateContainer,
                     Sofa.Component.Mass, Sofa.Component.SolidMechanics.Spring});
    }
};

} // namespace

/// A mass on a mapped state needs the acceleration and the mass term of the right-hand side to go through the
/// mapping: the particles move as with the same mass on their main state
TEST_F(EulerImplicitIntegrationSchemeMappedStates_test, massOnMappedStateMovesLikeMassOnMainState)
{
    MappedStatesScene massOnMain(false);
    MappedStatesScene massOnMapped(true);
    massOnMain.step(50);
    massOnMapped.step(50);

    const auto& x = massOnMain.positions(massOnMain.particles);
    const auto& y = massOnMapped.positions(massOnMapped.particles);
    ASSERT_EQ(x.size(), y.size());
    EXPECT_LT(x[0][1], -1e-3); // it fell
    for (std::size_t i = 0; i < x.size(); ++i)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(x[i][c], y[i][c], 1e-12) << "particle " << i << " coordinate " << c;
}

/// Without a mass on a mapped state, the acceleration is not propagated to the mapped states and the mass term of the
/// right-hand side is not pulled back from them: the mapping is applied once less in each direction per evaluation of
/// the right-hand side than with a mass on the mapped state
TEST_F(EulerImplicitIntegrationSchemeMappedStates_test, accelerationGoesThroughTheMappingsOnlyForMappedMasses)
{
    MappedStatesScene massOnMain(false);
    MappedStatesScene massOnMapped(true);
    const unsigned int mainJ0 = massOnMain.mapping->nbApplyJ, mainJT0 = massOnMain.mapping->nbApplyJT;
    const unsigned int mappedJ0 = massOnMapped.mapping->nbApplyJ, mappedJT0 = massOnMapped.mapping->nbApplyJT;
    constexpr int nbSteps = 20;
    massOnMain.step(nbSteps);
    massOnMapped.step(nbSteps);

    // one evaluation of the right-hand side per time step (EulerImplicitIntegrationScheme)
    EXPECT_EQ((massOnMapped.mapping->nbApplyJ - mappedJ0) - (massOnMain.mapping->nbApplyJ - mainJ0), 1u * nbSteps);
    EXPECT_EQ((massOnMapped.mapping->nbApplyJT - mappedJT0) - (massOnMain.mapping->nbApplyJT - mainJT0), 1u * nbSteps);
}

} // namespace sofa
