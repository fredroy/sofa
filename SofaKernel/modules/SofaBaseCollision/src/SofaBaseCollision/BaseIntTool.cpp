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
#define SOFA_SOFABASECOLLISION_BASEINTTOOL_DEFINITION

#include <SofaBaseCollision/BaseIntTool.h>
#include <SofaBaseCollision/CapsuleIntTool.h>
#include <SofaBaseCollision/OBBIntTool.h>

namespace sofa::component::collision
{

template<> SOFA_SOFABASECOLLISION_API
bool BaseIntTool<Cube, Cube>::testIntersection(Cube &cube1, Cube &cube2,SReal alarmDist)
{
    if (cube1 == cube2)
    {
        if (cube1.getConeAngle() < M_PI / 2)
            return false;
        else
            return true;
    }

    const auto& minVect1 = cube1.minVect();
    const auto& minVect2 = cube2.minVect();
    const auto& maxVect1 = cube1.maxVect();
    const auto& maxVect2 = cube2.maxVect();

    for (int i = 0; i < 3; i++)
    {
        if ( minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist )
            return false;
    }

    return true;
}

template<> SOFA_SOFABASECOLLISION_API
int BaseIntTool<Cube, Cube>::computeIntersection(Cube& c1, Cube& c2, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    SOFA_UNUSED(c1);
    SOFA_UNUSED(c2);
    SOFA_UNUSED(alarmDist);
    SOFA_UNUSED(contactDist);
    SOFA_UNUSED(contacts);

    return 0;
}

template <class DataTypes1, class DataTypes2>
bool BaseIntTool<TSphere<DataTypes1>, TSphere<DataTypes2>>::testIntersection(TSphere<DataTypes1>& sph1, TSphere<DataTypes2>& sph2, SReal alarmDist)
{
    auto r = sph1.r() + sph2.r() + alarmDist;
    return ( sph1.center() - sph2.center() ).norm2() <= r*r;
}

template <class DataTypes1, class DataTypes2>
int BaseIntTool<TSphere<DataTypes1>, TSphere<DataTypes2>>::computeIntersection(TSphere<DataTypes1>& sph1, TSphere<DataTypes2>& sph2, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    SReal r = sph1.r() + sph2.r();
    SReal myAlarmDist = alarmDist + r;
    defaulttype::Vector3 dist = sph2.center() - sph1.center();
    SReal norm2 = dist.norm2();
    
    if (norm2 > myAlarmDist*myAlarmDist)
        return 0;
    
    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    SReal distSph1Sph2 = helper::rsqrt(norm2);
    detection->normal = dist / distSph1Sph2;
    detection->point[0] = sph1.getContactPointByNormal( -detection->normal );
    detection->point[1] = sph2.getContactPointByNormal( detection->normal );
    
    detection->value = distSph1Sph2 - r - contactDist;
    detection->elem.first = sph1;
    detection->elem.second = sph2;
    detection->id = (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex();
    
    return 1;
}

template <class DataTypes1, class DataTypes2>
bool BaseIntTool<TCapsule<DataTypes1>, TCapsule<DataTypes2>>::testIntersection(TCapsule<DataTypes1>& c1, TCapsule<DataTypes2>& c2, SReal alarmDist)
{
    SOFA_UNUSED(c1);
    SOFA_UNUSED(c2);
    SOFA_UNUSED(alarmDist);

    return false;
}

template <class DataTypes1, class DataTypes2>
int BaseIntTool<TCapsule<DataTypes1>, TCapsule<DataTypes2>>::computeIntersection(TCapsule<DataTypes1>& c1, TCapsule<DataTypes2>& c2, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    return CapsuleIntTool::computeIntersection(c1, c2, alarmDist, contactDist, contacts);
}

template <class DataTypes1, class DataTypes2>
bool BaseIntTool<TCapsule<DataTypes1>, TSphere<DataTypes2>>::testIntersection(TCapsule<DataTypes1>& cap, TSphere<DataTypes2>& sph, SReal alarmDist)
{
    SOFA_UNUSED(cap);
    SOFA_UNUSED(sph);
    SOFA_UNUSED(alarmDist);

    return false;
}

template <class DataTypes1, class DataTypes2>
int BaseIntTool<TCapsule<DataTypes1>, TSphere<DataTypes2>>::computeIntersection(TCapsule<DataTypes1>& cap, TSphere<DataTypes2>& sph, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    return CapsuleIntTool::computeIntersection(cap, sph, alarmDist, contactDist, contacts);
}

template <class DataTypes>
bool BaseIntTool<TCapsule<DataTypes>, OBB>::testIntersection(TCapsule<DataTypes>& cap, OBB& obb, SReal alarmDist)
{
    SOFA_UNUSED(cap);
    SOFA_UNUSED(obb);
    SOFA_UNUSED(alarmDist);

    return false;
}

template <class DataTypes>
int BaseIntTool<TCapsule<DataTypes>, OBB>::computeIntersection(TCapsule<DataTypes>& cap, OBB& obb, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    return CapsuleIntTool::computeIntersection(cap, obb, alarmDist, contactDist, contacts);
}

template <class DataTypes>
bool BaseIntTool<TSphere<DataTypes>, OBB>::testIntersection(TSphere<DataTypes>& sph, OBB& obb, SReal alarmDist)
{
    SOFA_UNUSED(sph);
    SOFA_UNUSED(obb);
    SOFA_UNUSED(alarmDist);

    return false;
}

template <class DataTypes>
int BaseIntTool<TSphere<DataTypes>, OBB>::computeIntersection(TSphere<DataTypes>& sph, OBB& obb, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    return OBBIntTool::computeIntersection(sph, obb, alarmDist, contactDist, contacts);
}

template<>
bool BaseIntTool<OBB, OBB>::testIntersection(OBB& obb1, OBB& obb2, SReal alarmDist)
{
    SOFA_UNUSED(obb1);
    SOFA_UNUSED(obb2);
    SOFA_UNUSED(alarmDist);

    return false;
}

template<>
int BaseIntTool<OBB, OBB>::computeIntersection(OBB& obb1, OBB& obb2, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    return OBBIntTool::computeIntersection(obb1, obb2, alarmDist, contactDist, contacts);
}

template class SOFA_SOFABASECOLLISION_API BaseIntTool<Cube, Cube>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<OBB, OBB>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Vec3Types>, TSphere<defaulttype::Vec3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Rigid3Types>, TSphere<defaulttype::Rigid3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Vec3Types>, TSphere<defaulttype::Rigid3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Rigid3Types>, TSphere<defaulttype::Vec3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, TCapsule<defaulttype::Vec3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Rigid3Types>, TCapsule<defaulttype::Rigid3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, TCapsule<defaulttype::Rigid3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, TSphere<defaulttype::Vec3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Rigid3Types>, TSphere<defaulttype::Rigid3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, TSphere<defaulttype::Rigid3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Rigid3Types>, TSphere<defaulttype::Vec3Types>>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, OBB>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Rigid3Types>, OBB>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Vec3Types>, OBB>;
template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Rigid3Types>, OBB>;

} // namespace sofa::component::collision
