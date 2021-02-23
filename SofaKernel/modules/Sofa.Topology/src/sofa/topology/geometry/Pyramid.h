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

#include <sofa/topology/geometry/Point.h>
#include <sofa/type/stdtype/fixed_array.h>

namespace sofa::topology::geometry
{
    using PyramidID = Index;

    class Pyramid : public sofa::type::stdtype::fixed_array<PointID, 5>
    {
    public:
        Pyramid() : sofa::type::stdtype::fixed_array<PointID, 5>(InvalidID, InvalidID, InvalidID, InvalidID, InvalidID) {}
        Pyramid(PointID a, PointID b, PointID c, PointID d, PointID e) : sofa::type::stdtype::fixed_array<PointID, 5>(a, b, c, d, e) {}
    };

    inline static const Pyramid InvalidPyramid;
}
