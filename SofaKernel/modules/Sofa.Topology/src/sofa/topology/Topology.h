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

#include <sofa/topology/config.h>

#include <sofa/topology/TopologyElementType.h>
#include <sofa/topology/geometry/Point.h>
#include <sofa/topology/geometry/Edge.h>
#include <sofa/topology/geometry/Triangle.h>
#include <sofa/topology/geometry/Quad.h>
#include <sofa/topology/geometry/Pentahedron.h>
#include <sofa/topology/geometry/Tetrahedron.h>
#include <sofa/topology/geometry/Pyramid.h>
#include <sofa/topology/geometry/Hexahedron.h>

#include <climits>
#include <string>

namespace sofa::topology
{

class SOFA_TOPOLOGY_API Topology
{
public:
    /// compatibility
    using Index = sofa::Index;
    static constexpr Index InvalidID = sofa::InvalidID;

    using ElemID = geometry::ElemID;
    using PointID = geometry::PointID;
    using EdgeID = geometry::EdgeID;
    using TriangleID = geometry::TriangleID;
    using QuadID = geometry::QuadID;
    using TetraID = geometry::TetraID;
    using TetrahedronID = geometry::TetrahedronID;
    using HexaID = geometry::HexaID;
    using HexahedronID = geometry::HexahedronID;
    using PentahedronID = geometry::PentahedronID;
    using PentaID = geometry::PentaID;
    using PyramidID = geometry::PyramidID;

    using Edge = geometry::Edge;
    using Triangle = geometry::Triangle;
    using Quad = geometry::Quad;
    using Tetrahedron = geometry::Tetrahedron;
    using Tetra = geometry::Tetra;
    using Pentahedron = geometry::Pentahedron;
    using Penta = geometry::Penta;
    using Pyramid = geometry::Pyramid;
    using Hexahedron = geometry::Hexahedron;
    using Hexa = geometry::Hexa;

protected:
    Topology() {}
    ~Topology() {}
public:
};

} // namespace sofa::topology