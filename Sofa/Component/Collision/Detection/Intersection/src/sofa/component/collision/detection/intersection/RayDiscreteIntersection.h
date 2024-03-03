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
#include <sofa/component/collision/detection/intersection/config.h>

#include <sofa/core/collision/Intersection.h>

#include <sofa/component/collision/detection/intersection/DiscreteIntersection.h>
#include <sofa/component/collision/geometry/SphereModel.h>
#include <sofa/component/collision/geometry/PointModel.h>
#include <sofa/component/collision/geometry/LineModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/component/collision/geometry/RayModel.h>

namespace sofa::component::collision::detection::intersection
{
class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API RayDiscreteIntersection : public core::collision::BaseIntersector
{

    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    RayDiscreteIntersection(DiscreteIntersection* object, bool addSelf=true);

    template<class T> bool testIntersection(collision::geometry::Ray&, collision::geometry::TSphere<T>&);
    bool testIntersection(collision::geometry::Ray&, collision::geometry::Triangle&);

    template<class T> int computeIntersection(collision::geometry::Ray&, collision::geometry::TSphere<T>&, OutputVector*);
    int computeIntersection(collision::geometry::Ray&, collision::geometry::Triangle&, OutputVector*);
    
    template<class T> bool testIntersection(collision::geometry::Ray&, collision::geometry::TSphere<T>&, const core::collision::IntersectionParameters& params);
    bool testIntersection(collision::geometry::Ray&, collision::geometry::Triangle&, const core::collision::IntersectionParameters& params);

    template<class T> int computeIntersection(collision::geometry::Ray&, collision::geometry::TSphere<T>&, OutputVector*, const core::collision::IntersectionParameters& params);
    int computeIntersection(collision::geometry::Ray&, collision::geometry::Triangle&, OutputVector*, const core::collision::IntersectionParameters& params);

protected:

    DiscreteIntersection* intersection;

};

} //namespace sofa::component::collision::detection::intersection
