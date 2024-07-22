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
#include <sofa/component/engine/select/config.h>

#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::select
{

/**
 * This interface defines ROI features.
 */
template <class DataTypes>
class BaseROI : public core::DataEngine
{
public:
    using Inherit = core::DataEngine;
    SOFA_CLASS(SOFA_TEMPLATE(BaseROI,DataTypes), Inherit);

    using VecCoord = typename DataTypes::VecCoord;
    using Coord = typename DataTypes::Coord;
    using Real = typename DataTypes::Real;
    using CPos = typename DataTypes::CPos;

    using PointID = core::topology::BaseMeshTopology::PointID;
    using SetIndex = core::topology::BaseMeshTopology::SetIndex;
    using Edge = core::topology::BaseMeshTopology::Edge;
    using VecEdge = type::vector<Edge>;
    using Triangle = core::topology::BaseMeshTopology::Triangle;
    using VecTriangle = type::vector<Triangle>;
    using Quad = core::topology::BaseMeshTopology::Quad;
    using VecQuad = type::vector<Quad>;
    using Tetra = core::topology::BaseMeshTopology::Tetra;
    using VecTetra = type::vector<Tetra>;
    using Hexa = core::topology::BaseMeshTopology::Hexa;
    using VecHexa = type::vector<Hexa>;

public:
    void init() final;
    void doUpdate() override; //cannot set final because of the tests;
    void draw(const core::visual::VisualParams* vparams) final;

    virtual void roiInit() = 0;
    virtual bool roiDoUpdate() = 0;
    virtual void roiDraw(const core::visual::VisualParams* vparams) = 0;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (!arg->getAttribute("template"))
        {
            // only check if this template is correct if no template was given
            if (context->getMechanicalState() && dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == nullptr)
            {
                arg->logError(std::string("No mechanical state with the datatype '") + DataTypes::Name() +
                              "' found in the context node.");
                return false; // this template is not the same as the existing MechanicalState
            }
        }

        return BaseObject::canCreate(obj, context, arg);
    }

public:
    //Input
    Data<VecCoord> d_X0;
    Data<VecEdge > d_edges; ///< Edge Topology
    Data<VecTriangle > d_triangles; ///< Triangle Topology
    Data<VecQuad > d_quad; ///< Quad Topology
    Data<VecTetra > d_tetrahedra; ///< Tetrahedron Topology
    Data<VecHexa > d_hexahedra; ///< Hexahedron Topology
    Data<bool> d_computeEdges; ///< If true, will compute edge list and index list inside the ROI. (default = true)
    Data<bool> d_computeTriangles; ///< If true, will compute triangle list and index list inside the ROI. (default = true)
    Data<bool> d_computeQuad; ///< If true, will compute quad list and index list inside the ROI. (default = true)
    Data<bool> d_computeTetrahedra; ///< If true, will compute tetrahedra list and index list inside the ROI. (default = true)
    Data<bool> d_computeHexahedra; ///< If true, will compute hexahedra list and index list inside the ROI. (default = true)
    Data<bool> d_strict; ///< If true, an element is inside the box if all of its nodes are inside. If False, only the center point of the element is checked. (default = true)

    //Output
    Data<SetIndex> d_indices; ///< Indices of the points contained in the ROI
    Data<SetIndex> d_edgeIndices; ///< Indices of the edges contained in the ROI
    Data<SetIndex> d_triangleIndices; ///< Indices of the triangles contained in the ROI
    Data<SetIndex> d_quadIndices; ///< Indices of the quad contained in the ROI
    Data<SetIndex> d_tetrahedronIndices; ///< Indices of the tetrahedra contained in the ROI
    Data<SetIndex> d_hexahedronIndices; ///< Indices of the hexahedra contained in the ROI
    Data<VecCoord> d_pointsInROI; ///< Points contained in the ROI
    Data<VecEdge> d_edgesInROI; ///< Edges contained in the ROI
    Data<VecTriangle> d_trianglesInROI; ///< Triangles contained in the ROI
    Data<VecQuad> d_quadInROI; ///< Quad contained in the ROI
    Data<VecTetra> d_tetrahedraInROI; ///< Tetrahedra contained in the ROI
    Data<VecHexa> d_hexahedraInROI; ///< Hexahedra contained in the ROI
    Data< sofa::Size > d_nbIndices; ///< Number of selected indices

    Data<VecCoord > d_pointsOutROI; ///< Points not contained in the ROI
    Data<type::vector<Edge> > d_edgesOutROI; ///< Edges not contained in the ROI
    Data<type::vector<Triangle> > d_trianglesOutROI; ///< Triangles not contained in the ROI
    Data<type::vector<Quad> > d_quadsOutROI; ///< Quads not contained in the ROI
    Data<type::vector<Tetra> > d_tetrahedraOutROI; ///< Tetrahedra not contained in the ROI
    Data<type::vector<Hexa> > d_hexahedraOutROI; ///< Hexahedra not contained in the ROI
    Data<SetIndex> d_indicesOut; ///< Indices of the points not contained in the ROI
    Data<SetIndex> d_edgeOutIndices; ///< Indices of the edges not contained in the ROI
    Data<SetIndex> d_triangleOutIndices; ///< Indices of the triangles not contained in the ROI
    Data<SetIndex> d_quadOutIndices; ///< Indices of the quads not contained in the ROI
    Data<SetIndex> d_tetrahedronOutIndices; ///< Indices of the tetrahedra not contained in the ROI
    Data<SetIndex> d_hexahedronOutIndices; ///< Indices of the hexahedra not contained in the ROI

    //Parameter
    Data<bool> d_drawROI; ///< Draw the ROI. (default = false)
    Data<bool> d_drawPoints; ///< Draw Points. (default = false)
    Data<bool> d_drawEdges; ///< Draw Edges. (default = false)
    Data<bool> d_drawTriangles; ///< Draw Triangles. (default = false)
    Data<bool> d_drawTetrahedra; ///< Draw Tetrahedra. (default = false)
    Data<bool> d_drawHexahedra; ///< Draw Tetrahedra. (default = false)
    Data<bool> d_drawQuads; ///< Draw Quads. (default = false)
    Data<double> d_drawSize; ///< rendering size for ROI and topological elements
    Data<bool> d_doUpdate; ///< If true, updates the selection at the beginning of simulation steps. (default = true)
protected:
    BaseROI();
    ~BaseROI() override = default;

    bool isPointIn(const PointID pid);

    template <typename Element>
    bool isInROI(const Element & e);
    
    template <typename Element>
    bool isInStrictROI(const Element & e);

    // main function to implement from this interface
    virtual bool isPointInROI(const CPos& p) = 0;

    // special cases can be implemented by overriding those functions
    virtual bool isEdgeInROI(const Edge& e);
    virtual bool isEdgeInStrictROI(const Edge& e);
    virtual bool isTriangleInROI(const Triangle& t);
    virtual bool isTriangleInStrictROI(const Triangle& t);
    virtual bool isQuadInROI(const Quad& q);
    virtual bool isQuadInStrictROI(const Quad& q);
    virtual bool isTetrahedronInROI(const Tetra& t);
    virtual bool isTetrahedronInStrictROI(const Tetra& t);
    virtual bool isHexahedronInROI(const Hexa& t);
    virtual bool isHexahedronInStrictROI(const Hexa& t);
};

} // namespace sofa::component::engine::select
