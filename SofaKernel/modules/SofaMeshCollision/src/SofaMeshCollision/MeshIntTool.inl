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
#include <SofaMeshCollision/MeshIntTool.h>

namespace sofa::component::collision
{

template <class DataTypes>
int MeshIntToolUtil::doCapPointInt(TCapsule<DataTypes>& cap, const defaulttype::Vector3& q,SReal alarmDist,SReal contactDist,OutputVector* contacts){
    const defaulttype::Vector3 p1 = cap.point1();
    const defaulttype::Vector3 p2 = cap.point2();
    const defaulttype::Vector3 AB = p2-p1;
    const defaulttype::Vector3 AQ = q -p1;
    SReal A;
    SReal b;
    A = AB*AB;
    b = AQ*AB;
    SReal cap_rad = cap.radius();

    SReal alpha = 0.5;

    alpha = b/A;//projection of the point on the capsule segment such as the projected point P = p1 + AB * alpha
    if (alpha < 0.0) alpha = 0.0;//if the projection is out the segment, we associate it to a segment apex
    else if (alpha > 1.0) alpha = 1.0;

    defaulttype::Vector3 p,pq;
    p = p1 + AB * alpha;
    pq = q-p;

    SReal enough_to_touch = alarmDist + cap_rad;
    if (pq.norm2() >= enough_to_touch * enough_to_touch)
        return 0;

    //const SReal contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    core::collision::DetectionOutput *detection = &*(contacts->end()-1);

    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal = pq;

    detection->value = detection->normal.norm();
    detection->normal /= detection->value;

    detection->value -= (contactDist + cap_rad);

    return 1;
}

template <class DataTypes>
int MeshIntToolUtil::doCapLineInt(TCapsule<DataTypes> & cap,const defaulttype::Vector3 & q1,const defaulttype::Vector3 & q2 ,SReal alarmDist,SReal contactDist,OutputVector* contacts, bool ignore_p1, bool ignore_p2)
{
    SReal cap_rad = cap.radius();
    const defaulttype::Vector3 p1 = cap.point1();
    const defaulttype::Vector3 p2 = cap.point2();

    return doCapLineInt(p1,p2,cap_rad,q1,q2,alarmDist,contactDist,contacts,ignore_p1,ignore_p2);
}

} // namespace sofa::component::collision
