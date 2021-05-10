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
#include <SofaBaseCollision/config.h>

#include <sofa/core/collision/DetectionOutput.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaBaseCollision/Sphere.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <SofaBaseCollision/OBBModel.h>


namespace sofa::component::collision
{

template <typename ElementType1, typename ElementType2>
class SOFA_SOFABASECOLLISION_API BaseIntTool
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(ElementType1&, ElementType2&, SReal alarmDist)
    {
        SOFA_UNUSED(alarmDist);

        msg_error("BaseIntTool") << "testIntersection not implemented for these elements.";
        return false;
    }

    static int computeIntersection(ElementType1&, ElementType2&, SReal alarmDist, SReal contactDist, OutputVector* contacts)
    {
        SOFA_UNUSED(alarmDist);
        SOFA_UNUSED(contactDist);
        SOFA_UNUSED(contacts);

        msg_error("BaseIntTool") << "computeIntersection not implemented for these elements.";
        return 0;
    }
};

template <class DataTypes1, class DataTypes2>
class SOFA_SOFABASECOLLISION_API BaseIntTool< TSphere<DataTypes1>, TSphere<DataTypes2> >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TSphere<DataTypes1>&, TSphere<DataTypes2>&, SReal alarmDist);
    static int computeIntersection(TSphere<DataTypes1>&, TSphere<DataTypes2>&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class DataTypes1, class DataTypes2>
class SOFA_SOFABASECOLLISION_API BaseIntTool< TCapsule<DataTypes1>, TCapsule<DataTypes2> >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TCapsule<DataTypes1>&, TCapsule<DataTypes2>&, SReal alarmDist);
    static int computeIntersection(TCapsule<DataTypes1>&, TCapsule<DataTypes2>&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class DataTypes1, class DataTypes2>
class SOFA_SOFABASECOLLISION_API BaseIntTool< TCapsule<DataTypes1>, TSphere<DataTypes2> >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TCapsule<DataTypes1>&, TSphere<DataTypes2>&, SReal alarmDist);
    static int computeIntersection(TCapsule<DataTypes1>&, TSphere<DataTypes2>&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class DataTypes>
class SOFA_SOFABASECOLLISION_API BaseIntTool< TCapsule<DataTypes>, OBB >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TCapsule<DataTypes>&, OBB&, SReal alarmDist);
    static int computeIntersection(TCapsule<DataTypes>&, OBB&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

template <class DataTypes>
class SOFA_SOFABASECOLLISION_API BaseIntTool< TSphere<DataTypes>, OBB >
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static bool testIntersection(TSphere<DataTypes>&, OBB&, SReal alarmDist);
    static int computeIntersection(TSphere<DataTypes>&, OBB&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
};

#ifndef SOFA_SOFABASECOLLISION_BASEINTTOOL_DEFINTION
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<Cube, Cube>;
extern template bool  SOFA_SOFABASECOLLISION_API BaseIntTool<Cube, Cube>::testIntersection(Cube&, Cube&, SReal alarmDist);
extern template int   SOFA_SOFABASECOLLISION_API BaseIntTool<Cube, Cube>::computeIntersection(Cube&, Cube&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<OBB, OBB>;
extern template bool  SOFA_SOFABASECOLLISION_API BaseIntTool<OBB, OBB>::testIntersection(OBB&, OBB&, SReal alarmDist);
extern template int   SOFA_SOFABASECOLLISION_API BaseIntTool<OBB, OBB>::computeIntersection(OBB&, OBB&, SReal alarmDist, SReal contactDist, OutputVector* contacts);
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Vec3Types>, TSphere<defaulttype::Vec3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Rigid3Types>, TSphere<defaulttype::Rigid3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Vec3Types>, TSphere<defaulttype::Rigid3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, TCapsule<defaulttype::Vec3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Rigid3Types>, TCapsule<defaulttype::Rigid3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, TCapsule<defaulttype::Rigid3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, TSphere<defaulttype::Vec3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Rigid3Types>, TSphere<defaulttype::Rigid3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, TSphere<defaulttype::Rigid3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Rigid3Types>, TSphere<defaulttype::Vec3Types>>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Vec3Types>, OBB>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TCapsule<defaulttype::Rigid3Types>, OBB>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Vec3Types>, OBB>;
extern template class SOFA_SOFABASECOLLISION_API BaseIntTool<TSphere<defaulttype::Rigid3Types>, OBB>;
#endif // SOFA_SOFABASECOLLISION_BASEINTTOOL_DEFINTION

} // namespace sofa::component::collision
