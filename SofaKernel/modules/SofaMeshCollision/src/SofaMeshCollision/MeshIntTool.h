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
#include <SofaMeshCollision/config.h>

#include <sofa/core/collision/Intersection.h>
#include <SofaBaseCollision/BaseIntTool.h>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <SofaBaseCollision/RigidCapsuleModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/IntrTriangleOBB.h>
#include <SofaBaseCollision/SphereModel.h>

namespace sofa::component::collision
{

class SOFA_SOFAMESHCOLLISION_API MeshIntToolUtil
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id
    template <class DataTypes>
    static int doCapPointInt(TCapsule<DataTypes>& cap, const defaulttype::Vector3& q, SReal alarmDist, SReal contactDist, OutputVector* contacts);
    ////!\ CAUTION : uninitialized fields detection->elem and detection->id
    template <class DataTypes>
    static int doCapLineInt(TCapsule<DataTypes>& cap, const defaulttype::Vector3& q1, const defaulttype::Vector3& q2, SReal alarmDist, SReal contactDist, OutputVector* contacts, bool ignore_p1 = false, bool ignore_p2 = false);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value
    static int doCapLineInt(const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, SReal cap_rad,
        const defaulttype::Vector3& q1, const defaulttype::Vector3& q2, SReal alarmDist, SReal contactDist, OutputVector* contacts, bool ignore_p1 = false, bool ignore_p2 = false);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value, you have to substract contactDist, because
    ///this function can be used also as doIntersectionTriangleSphere where the contactDist = getContactDist() + sphere_radius
    static int doIntersectionTrianglePoint(SReal dist2, int flags, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& p3, const defaulttype::Vector3& q, OutputVector* contacts, bool swapElems = false);

    //returns barycentric coords in alpha and beta so that to_be_projected = (1 - alpha - beta) * p1 + alpha * p2 + beta * p3
    static void triangleBaryCoords(const defaulttype::Vector3& to_be_projected, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& p3, SReal& alpha, SReal& beta);

    //flags are the flags of the Triangle and p1 p2 p3 its vertices, to_be_projected is the point to be projected on the triangle, i.e.
    //after this method, it will probably be different
    static int projectPointOnTriangle(int flags, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& p3, defaulttype::Vector3& to_be_projected);

    static int computeIntersection(Triangle& tri, int flags, OBB& obb, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};


template <class DataTypes>
class SOFA_SOFAMESHCOLLISION_API BaseIntTool< TCapsule<DataTypes>, Point >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TCapsule<DataTypes>&, Point&, SReal alarmDist);
    static int computeIntersection(TCapsule<DataTypes>&, Point&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class DataTypes>
class SOFA_SOFAMESHCOLLISION_API BaseIntTool< TCapsule<DataTypes>, Line >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TCapsule<DataTypes>&, Line&, SReal alarmDist);
    static int computeIntersection(TCapsule<DataTypes>&, Line&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class DataTypes>
class SOFA_SOFAMESHCOLLISION_API BaseIntTool< TCapsule<DataTypes>, Triangle >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TCapsule<DataTypes>&, Triangle&, SReal alarmDist);
    static int computeIntersection(TCapsule<DataTypes>&, Triangle&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class DataTypes>
class SOFA_SOFAMESHCOLLISION_API BaseIntTool< TSphere<DataTypes>, Point >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TSphere<DataTypes>&, Point&, SReal alarmDist);
    static int computeIntersection(TSphere<DataTypes>&, Point&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class DataTypes>
class SOFA_SOFAMESHCOLLISION_API BaseIntTool< Line, TSphere<DataTypes> >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(Line&, TSphere<DataTypes>&, SReal alarmDist);
    static int computeIntersection(Line&, TSphere<DataTypes>&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class DataTypes>
class SOFA_SOFAMESHCOLLISION_API BaseIntTool< Triangle, TSphere<DataTypes> >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(Triangle&, TSphere<DataTypes>&, SReal alarmDist);
    static int computeIntersection(Triangle&, TSphere<DataTypes>&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class TReal>
class SOFA_SOFAMESHCOLLISION_API BaseIntTool< TSphere<defaulttype::StdVectorTypes<defaulttype::Vec<3, TReal>, defaulttype::Vec<3, TReal>, TReal> >, Point >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TSphere<defaulttype::StdVectorTypes<defaulttype::Vec<3, TReal>, defaulttype::Vec<3, TReal>, TReal> >&, Point&, SReal alarmDist);
    static int computeIntersection(TSphere<defaulttype::StdVectorTypes<defaulttype::Vec<3, TReal>, defaulttype::Vec<3, TReal>, TReal> >&, Point&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class TReal>
class SOFA_SOFAMESHCOLLISION_API BaseIntTool< Line, TSphere<defaulttype::StdVectorTypes<defaulttype::Vec<3, TReal>, defaulttype::Vec<3, TReal>, TReal> > >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(Line&, TSphere<defaulttype::StdVectorTypes<defaulttype::Vec<3, TReal>, defaulttype::Vec<3, TReal>, TReal> >&, SReal alarmDist);
    static int computeIntersection(Line&, TSphere<defaulttype::StdVectorTypes<defaulttype::Vec<3, TReal>, defaulttype::Vec<3, TReal>, TReal> >&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class TReal>
class SOFA_SOFAMESHCOLLISION_API BaseIntTool< Triangle, TSphere<defaulttype::StdVectorTypes<defaulttype::Vec<3, TReal>, defaulttype::Vec<3, TReal>, TReal> > >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(Triangle&, TSphere<defaulttype::StdVectorTypes<defaulttype::Vec<3, TReal>, defaulttype::Vec<3, TReal>, TReal> >&, SReal alarmDist);
    static int computeIntersection(Triangle&, TSphere<defaulttype::StdVectorTypes<defaulttype::Vec<3, TReal>, defaulttype::Vec<3, TReal>, TReal> >&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};


#if  !defined(SOFA_SOFAMESHCOLLISION_MESHINTTOOL_DEFINITION)
extern template SOFA_SOFAMESHCOLLISION_API int MeshIntToolUtil::doCapPointInt(TCapsule<sofa::defaulttype::Vec3Types>& cap, const sofa::defaulttype::Vector3& q,SReal alarmDist,SReal contactDist,OutputVector* contacts);
extern template SOFA_SOFAMESHCOLLISION_API int MeshIntToolUtil::doCapLineInt(TCapsule<sofa::defaulttype::Vec3Types>& cap, const sofa::defaulttype::Vector3& q1, const sofa::defaulttype::Vector3& q2, SReal alarmDist, SReal contactDist, OutputVector* contacts, bool ignore_p1, bool ignore_p2);
extern template SOFA_SOFAMESHCOLLISION_API int MeshIntToolUtil::doCapPointInt(TCapsule<sofa::defaulttype::Rigid3Types>& cap, const sofa::defaulttype::Vector3& q, SReal alarmDist, SReal contactDist, OutputVector* contacts);
extern template SOFA_SOFAMESHCOLLISION_API int MeshIntToolUtil::doCapLineInt(TCapsule<sofa::defaulttype::Rigid3Types>& cap, const sofa::defaulttype::Vector3& q1, const sofa::defaulttype::Vector3& q2, SReal alarmDist, SReal contactDist, OutputVector* contacts, bool ignore_p1, bool ignore_p2);


extern template SOFA_SOFAMESHCOLLISION_API bool BaseIntTool<Triangle, OBB>::testIntersection(Triangle& tri, OBB& obb, SReal alarmDist);
extern template SOFA_SOFAMESHCOLLISION_API int BaseIntTool<Triangle, OBB>::computeIntersection(Triangle& tri, OBB& obb, SReal alarmDist, SReal contactDist, OutputVector* contacts);

extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Vec3Types>, Point>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Vec3Types>, Line>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Vec3Types>, Triangle>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Rigid3Types>, Point>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Rigid3Types>, Line>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Rigid3Types>, Triangle>;

extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TSphere<sofa::defaulttype::Vec3Types>, Point>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<Line, TSphere<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<Triangle, TSphere<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TSphere<sofa::defaulttype::Rigid3Types>, Point>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<Line, TSphere<sofa::defaulttype::Rigid3Types>>;
extern template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<Triangle, TSphere<sofa::defaulttype::Rigid3Types>>;

#endif


} // namespace sofa::component::collision

