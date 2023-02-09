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
#include <gtest/gtest.h>
#include <sofa/component/engine/transform/DisplacementMatrixEngine.h>

namespace sofa
{
TEST(DisplacementTransformEngine, getTemplateName)
{
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::component::engine::transform::DisplacementTransformEngine<sofa::defaulttype::Rigid3Types, sofa::type::Mat4x4>
        >();

        if constexpr (std::is_same_v<SReal, double>)
        {
            EXPECT_EQ(engine->getTemplateName(), "Rigid3d,Mat4x4d");
        }
        else
        {
            EXPECT_EQ(engine->getTemplateName(), "Rigid3f,Mat4x4f");
        }
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::component::engine::transform::DisplacementTransformEngine<sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types::Coord>
        >();

        if constexpr (std::is_same_v<SReal, double>)
        {
            EXPECT_EQ(engine->getTemplateName(), "Rigid3d,RigidCoord3d");
        }
        else
        {
            EXPECT_EQ(engine->getTemplateName(), "Rigid3f,RigidCoord3f");
        }
    }
}
}
