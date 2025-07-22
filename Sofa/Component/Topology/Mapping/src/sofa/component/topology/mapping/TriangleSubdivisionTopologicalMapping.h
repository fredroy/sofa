/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/component/topology/mapping/config.h>

#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/type/Vec.h>
#include <map>
#include <set>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/topology/TopologyData.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::topology::mapping
{

using namespace sofa::defaulttype;
using namespace sofa::core::topology;
using sofa::core::topology::BaseMeshTopology;
using namespace sofa::core;

/**
 * This class, called TriangleSubdivisionTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = TriangleSetTopology
 * OUTPUT TOPOLOGY = subdivided TriangleSetTopology
 *
 * Each triangle in the input Topology will be recursively subdivided in output topology
 *
 * TriangleSubdivisionTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
 *
*/

class SOFA_COMPONENT_TOPOLOGY_MAPPING_API TriangleSubdivisionTopologicalMapping : public TopologicalMapping, public sofa::core::DataEngine
{
public:
    SOFA_CLASS(TriangleSubdivisionTopologicalMapping,TopologicalMapping);
    
    template<typename T>
    using Data = sofa::Data<T>;

    typedef BaseMeshTopology::Edge            Edge;
    typedef BaseMeshTopology::Triangle        Triangle;
    typedef BaseMeshTopology::SeqEdges        SeqEdges;
    typedef BaseMeshTopology::SeqTriangles        SeqTriangles;

    typedef sofa::defaulttype::Vec3Types                   InitTypes;
    typedef sofa::defaulttype::Vec3Types::Coord            Coord;
    typedef sofa::defaulttype::Vec3Types::VecCoord         VecCoord;

    typedef sofa::defaulttype::RigidTypes::VecCoord InVecCoord;

    /** \brief Initializes the target BaseTopology from the source BaseTopology.
     */
    virtual void init() override;

    /** \brief Translates the TopologyChange objects from the source to the target.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the first topology changes on the second topology.
     *
     */
    virtual void updateTopologicalMappingTopDown() override;

    /** \brief Translates the TopologyChange objects from the target to the source.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the second topology changes on the first topology.
     *
     */
    virtual void updateTopologicalMappingBottomUp() override;


    void doUpdate() override;

    const SeqEdges& getSubEdges() const { return m_subEdges; }

    // Create this mapping as an engine
    // inputs
    Data <VecCoord> d_inputPositions;
    Data <sofa::type::vector <Triangle> > d_inputTriangles;

    // outputs
    Data <VecCoord> d_outputPositions;
    Data <sofa::type::vector <Triangle> > d_outputTriangles;

protected:
    /** \brief Constructor.
     *
     */
    TriangleSubdivisionTopologicalMapping();

    /** \brief Destructor.
     */
    virtual ~TriangleSubdivisionTopologicalMapping() = default;

    Data<int> d_subdivisions;

    SeqEdges m_subEdges;
    SeqTriangles m_innerTriangles; // for a triangle [a][b][c], this triangle is [a'][b'][c'] with a' middle of [ab], b' of [bc] and c' of [ca]

    void subdivide(InitTypes::VecCoord &subVertices, const SeqTriangles subTriangles, SeqTriangles &newSubTriangles);
    void subdivide(InitTypes::VecCoord &subVertices, const SeqEdges subEdges, SeqEdges &newSubEdges);
    void addVertexAndFindIndex(InitTypes::VecCoord &subVertices, const Coord &vertex, int &index);

    void subdivideVertices ();
    void subdivideTriangles ();
    void propagateInformation (); //Should disappear with data engines
};

} // namespace sofa::component::topology::mapping
