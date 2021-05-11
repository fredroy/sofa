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
#define SOFA_SOFAMESHCOLLISION_MESHINTTOOL_DEFINITION
#include <SofaMeshCollision/MeshIntTool.inl>

namespace sofa::component::collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

template<> SOFA_SOFAMESHCOLLISION_API
bool BaseIntTool<Triangle, OBB>::testIntersection(Triangle& tri, OBB& obb, SReal alarmDist)
{
    SOFA_UNUSED(tri);
    SOFA_UNUSED(obb);
    SOFA_UNUSED(alarmDist);

    return false;
}

template <> SOFA_SOFAMESHCOLLISION_API
int BaseIntTool<Triangle, OBB>::computeIntersection(Triangle& tri, OBB& obb, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    return MeshIntToolUtil::computeIntersection(tri, tri.flags(), obb, alarmDist, contactDist, contacts);
}

template <class DataTypes>
bool BaseIntTool<TCapsule<DataTypes>, Point>::testIntersection(TCapsule<DataTypes>& cap, Point& pnt, SReal alarmDist)
{
    SOFA_UNUSED(cap);
    SOFA_UNUSED(pnt);
    SOFA_UNUSED(alarmDist);

    return false;
}

template <class DataTypes>
int BaseIntTool<TCapsule<DataTypes>, Point>::computeIntersection(TCapsule<DataTypes>& cap, Point& pnt, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    if (MeshIntToolUtil::doCapPointInt(cap, pnt.p(), alarmDist, contactDist, contacts)) 
    {
        DetectionOutput* detection = &*(contacts->end() - 1);

        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, pnt);

        return 1;
    }

    return 0;
}

template <class DataTypes>
bool BaseIntTool<TCapsule<DataTypes>, Line>::testIntersection(TCapsule<DataTypes>& cap, Line& lin, SReal alarmDist)
{
    SOFA_UNUSED(cap);
    SOFA_UNUSED(lin);
    SOFA_UNUSED(alarmDist);

    return true;
}

template <class DataTypes>
int BaseIntTool<TCapsule<DataTypes>, Line>::computeIntersection(TCapsule<DataTypes>& cap, Line& lin, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    SReal cap_rad = cap.radius();
    const defaulttype::Vector3 p1 = cap.point1();
    const defaulttype::Vector3 p2 = cap.point2();
    const defaulttype::Vector3 q1 = lin.p1();
    const defaulttype::Vector3 q2 = lin.p2();

    if (MeshIntToolUtil::doCapLineInt(p1, p2, cap_rad, q1, q2, alarmDist, contactDist, contacts)) {
        OutputVector::iterator detection = contacts->end() - 1;
        //detection->id = cap.getCollisionModel()->getSize() > lin.getCollisionModel()->getSize() ? cap.getIndex() : lin.getIndex();
        detection->id = cap.getIndex();
        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, lin);
        return 1;
    }

    return 0;
}

template <class DataTypes>
bool BaseIntTool<TCapsule<DataTypes>, Triangle>::testIntersection(TCapsule<DataTypes>& cap, Triangle& tri, SReal alarmDist)
{
    SOFA_UNUSED(cap);
    SOFA_UNUSED(tri);
    SOFA_UNUSED(alarmDist);

    return true;
}

template <class DataTypes>
int BaseIntTool<TCapsule<DataTypes>, Triangle>::computeIntersection(TCapsule<DataTypes>& cap, Triangle& tri, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    const int tri_flg = tri.flags();

    int id = cap.getIndex();
    int n = 0;

    const defaulttype::Vector3 cap_p1 = cap.point1();
    const defaulttype::Vector3 cap_p2 = cap.point2();
    SReal cap_rad = cap.radius();
    SReal dist2 = (alarmDist + cap_rad) * (alarmDist + cap_rad);

    const defaulttype::Vector3 tri_p1 = tri.p1();
    const defaulttype::Vector3 tri_p2 = tri.p2();
    const defaulttype::Vector3 tri_p3 = tri.p3();

    SReal substract_dist = contactDist + cap_rad;
    n += MeshIntToolUtil::doIntersectionTrianglePoint(dist2, tri_flg, tri_p1, tri_p2, tri_p3, cap_p1, contacts, true);
    n += MeshIntToolUtil::doIntersectionTrianglePoint(dist2, tri_flg, tri_p1, tri_p2, tri_p3, cap_p2, contacts, true);

    if (n == 2) {
        OutputVector::iterator detection1 = contacts->end() - 2;
        OutputVector::iterator detection2 = contacts->end() - 1;

        if (detection1->value > detection2->value - 1e-15 && detection1->value < detection2->value + 1e-15) {
            detection1->point[0] = (detection1->point[0] + detection2->point[0]) / 2.0;
            detection1->point[1] = (detection1->point[1] + detection2->point[1]) / 2.0;
            detection1->normal = (detection1->normal + detection2->normal) / 2.0;
            detection1->value = (detection1->value + detection2->value) / 2.0 - substract_dist;
            detection1->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);

            contacts->pop_back();
            n = 1;
        }
        else {
            for (OutputVector::iterator detection = contacts->end() - n; detection != contacts->end(); ++detection) {
                detection->value -= substract_dist;
                detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
                detection->id = id;
            }
        }
    }
    else {
        for (OutputVector::iterator detection = contacts->end() - n; detection != contacts->end(); ++detection) {
            detection->value -= substract_dist;
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
            detection->id = id;
        }
    }

    int old_n = n;
    n = 0;

    if (tri_flg & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)
        n += MeshIntToolUtil::doCapLineInt(cap_p1, cap_p2, cap_rad, tri_p1, tri_p2, alarmDist, contactDist, contacts, !(tri_flg & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1), !(tri_flg & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2));
    if (tri_flg & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E23)
        n += MeshIntToolUtil::doCapLineInt(cap_p1, cap_p2, cap_rad, tri_p2, tri_p3, alarmDist, contactDist, contacts, !(tri_flg & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2), !(tri_flg & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3));
    if (tri_flg & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)
        n += MeshIntToolUtil::doCapLineInt(cap_p1, cap_p2, cap_rad, tri_p3, tri_p1, alarmDist, contactDist, contacts, !(tri_flg & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3), !(tri_flg & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1));

    for (OutputVector::iterator detection = contacts->end() - n; detection != contacts->end(); ++detection) {
        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
        detection->id = id;
    }

    return n + old_n;
}

template <class DataTypes>
bool BaseIntTool<TSphere<DataTypes>, Point>::testIntersection(TSphere<DataTypes>& sph, Point& pnt, SReal alarmDist)
{
    SOFA_UNUSED(sph);
    SOFA_UNUSED(pnt);
    SOFA_UNUSED(alarmDist);

    return false;
}

template <class DataTypes>
int BaseIntTool<TSphere<DataTypes>, Point>::computeIntersection(TSphere<DataTypes>& sph, Point& pnt, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    const typename DataTypes::Real myAlarmDist = alarmDist + sph.r();

    const auto& P = sph.center();
    const auto& Q = pnt.p();
    auto PQ = Q - P;
    if (PQ.norm2() >= myAlarmDist * myAlarmDist)
        return 0;

    const auto myContactDist = contactDist + sph.r();

    contacts->resize(contacts->size() + 1);
    DetectionOutput* detection = &*(contacts->end() - 1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(sph, pnt);
    detection->id = (sph.getCollisionModel()->getSize() > pnt.getCollisionModel()->getSize()) ? sph.getIndex() : pnt.getIndex();
    detection->point[1] = Q;
    detection->normal = PQ;
    detection->value = detection->normal.norm();
    if (detection->value > 1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        detection->normal = typename DataTypes::CPos(1, 0, 0);
    }
    detection->point[0] = sph.getContactPointByNormal(-detection->normal);

    detection->value -= myContactDist;
    return 1;
}

template <class DataTypes>
bool BaseIntTool<Line, TSphere<DataTypes>>::testIntersection(Line& lin, TSphere<DataTypes>& sph, SReal alarmDist)
{
    SOFA_UNUSED(sph);
    SOFA_UNUSED(lin);
    SOFA_UNUSED(alarmDist);

    return false;
}

template <class DataTypes>
int BaseIntTool<Line, TSphere<DataTypes >>::computeIntersection(Line& lin, TSphere<DataTypes>& sph, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    const typename DataTypes::Real myAlarmDist = alarmDist + sph.r();

    const auto x32 = lin.p1() - lin.p2();
    const auto x31 = sph.center() - lin.p2();

    typename DataTypes::Real A;
    typename DataTypes::Real b;
    A = x32 * x32;
    b = x32 * x31;

    typename DataTypes::Real alpha = 0.5;
    auto Q = lin.p1() - x32 * alpha;

    if (alpha <= 0) {
        Q = lin.p1();
    }
    else if (alpha >= 1) {
        Q = lin.p2();
    }

    const auto& P = sph.center();
    const auto& QP = P - Q;

    if (QP.norm2() >= myAlarmDist * myAlarmDist)
        return 0;

    const typename DataTypes::Real myContactDist = contactDist + sph.r();

    contacts->resize(contacts->size() + 1);
    DetectionOutput* detection = &*(contacts->end() - 1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(lin, sph);
    detection->id = sph.getIndex();
    detection->point[0] = Q;
    detection->normal = QP;
    detection->value = detection->normal.norm();
    if (detection->value > 1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        detection->normal = typename DataTypes::CPos(1, 0, 0);
    }
    detection->point[1] = sph.getContactPointByNormal(detection->normal);
    detection->value -= myContactDist;
    return 1;
}

template <class DataTypes>
bool BaseIntTool<Triangle, TSphere<DataTypes>>::testIntersection(Triangle& tri, TSphere<DataTypes>& sph, SReal alarmDist)
{
    const auto x13 = tri.p1() - tri.p2();
    const auto x23 = tri.p1() - tri.p3();
    const auto x03 = tri.p1() - sph.center();
    defaulttype::Matrix2 A;
    defaulttype::Vector2 b;
    A[0][0] = x13 * x13;
    A[1][1] = x23 * x23;
    A[0][1] = A[1][0] = x13 * x23;
    b[0] = x13 * x03;
    b[1] = x23 * x03;
    const SReal det = defaulttype::determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    //if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0] * A[1][1] - b[1] * A[0][1]) / det;
        beta = (b[1] * A[0][0] - b[0] * A[1][0]) / det;
        if (alpha < 0.000001 ||
            beta < 0.000001 ||
            alpha + beta  > 0.999999)
            return false;
    }

    defaulttype::Vector3 P, Q, PQ;
    P = sph.center();
    Q = tri.p1() - x13 * alpha - x23 * beta;
    PQ = Q - P;

    if (PQ.norm2() < alarmDist * alarmDist)
    {
        return true;
    }
    else
        return false;
}

template <class DataTypes>
int BaseIntTool<Triangle, TSphere<DataTypes >>::computeIntersection(Triangle& tri, TSphere<DataTypes>& sph, SReal alarmDist, SReal contactDist, OutputVector* contacts)
{
    const auto& sph_center = sph.p();
    auto proj_p = sph_center;
    if (MeshIntToolUtil::projectPointOnTriangle(tri.flags(), tri.p1(), tri.p2(), tri.p3(), proj_p)) {

        const auto proj_p_sph_center = sph_center - proj_p;
        typename DataTypes::Real myAlarmDist = alarmDist + sph.r();
        if (proj_p_sph_center.norm2() >= myAlarmDist * myAlarmDist)
            return 0;

        contacts->resize(contacts->size() + 1);
        DetectionOutput* detection = &*(contacts->end() - 1);
        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(tri, sph);
        detection->id = sph.getIndex();
        detection->point[0] = proj_p;
        detection->normal = proj_p_sph_center;
        detection->value = detection->normal.norm();
        detection->normal /= detection->value;
        detection->point[1] = sph.getContactPointByNormal(detection->normal);
        detection->value -= (contactDist + sph.r());
    }
    else {
        return 0;
    }
}

int MeshIntToolUtil::doCapLineInt(const Vector3 & p1,const Vector3 & p2,SReal cap_rad,
                         const Vector3 & q1, const Vector3 & q2,SReal alarmDist,SReal contactDist,OutputVector *contacts, bool ignore_p1, bool ignore_p2){
    const Vector3 AB = p2-p1;//capsule segment
    const Vector3 CD = q2-q1;//line segment
    const Vector3 AC = q1-p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;
    const SReal det = defaulttype::determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    if (det < -0.000000000001 || det > 0.000000000001)//AB and CD are not on the same plane
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;

        if(alpha < 0)
            alpha = 0;
        else if(alpha > 1)
            alpha = 1;

        if(beta < 0)
            beta = 0;
        else if(beta > 1)
            beta = 1;
    }
    else{//Segments on a same plane. Here the idea to find the nearest points
        //is to project segment apexes on the other segment.
        //Visual example with semgents AB and CD :
        //            A----------------B
        //                     C----------------D
        //After projection :
        //            A--------c-------B
        //                     C-------b--------D
        //So the nearest points are p and q which are respecively in the middle of cB and Cb:
        //            A--------c---p---B
        //                     C---q---b--------D

        Vector3 AD = q2 - p1;
        Vector3 CB = p2 - q1;

        SReal AB_norm2 = AB.norm2();
        SReal CD_norm2 = CD.norm2();
        SReal c_proj= b[0]/AB_norm2;//alpha = (AB * AC)/AB_norm2
        SReal d_proj = (AB * AD)/AB_norm2;
        SReal a_proj = b[1];//beta = (-CD*AC)/CD_norm2
        SReal b_proj= (CD*CB)/CD_norm2;

        if(c_proj >= 0 && c_proj <= 1){//projection of C on AB is lying on AB
            if(d_proj > 1){//case :
                           //             A----------------B
                           //                      C---------------D
                alpha = (1.0 + c_proj)/2.0;
                beta = b_proj/2.0;
            }
            else if(d_proj < 0){//case :
                                //             A----------------B
                                //     D----------------C
                alpha = c_proj/2.0;
                beta = (1 + a_proj)/2.0;
            }
            else{//case :
                //             A----------------B
                //                 C------D
                alpha = (c_proj + d_proj)/2.0;
                beta  = 0.5;
            }
        }
        else if(d_proj >= 0 && d_proj <= 1){
            if(c_proj < 0){//case :
                           //             A----------------B
                           //     C----------------D
                alpha = d_proj /2.0;
                beta = (1 + a_proj)/2.0;
            }
            else{//case :
                 //          A---------------B
                 //                 D-------------C
                alpha = (1 + d_proj)/2.0;
                beta = b_proj/2.0;
            }
        }
        else{
            if(c_proj * d_proj < 0){//case :
                                    //           A--------B
                                    //       D-----------------C
                alpha = 0.5;
                beta = (a_proj + b_proj)/2.0;
            }
            else{
                if(c_proj < 0){//case :
                               //                    A---------------B
                               // C-------------D

                    alpha = 0;
                }
                else{
                    alpha = 1;
                }

                if(a_proj < 0){//case :
                               // A---------------B
                               //                     C-------------D
                }
                else{//case :
                     //                     A---------------B
                     //   C-------------D
                    beta = 1;
                }
            }
        }
    }

    if(ignore_p1 && beta == 0)
        return 0;
    if(ignore_p2 && beta == 1)
        return 0;

    SReal enough_to_touch = alarmDist + cap_rad;
    Vector3 p,q,pq;
    p = p1 + AB * alpha;
    q = q1 + CD * beta;
    pq = q-p;
    if (pq.norm2() >= enough_to_touch*enough_to_touch)
        return 0;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal=pq;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= (contactDist + cap_rad);

    ///!\ CAUTION : uninitialized fields detection->elem and detection->id

    return 1;
}


int MeshIntToolUtil::doIntersectionTrianglePoint(SReal dist2, int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3,const Vector3& q, OutputVector* contacts,bool swapElems)
{
    const Vector3 AB = p2-p1;
    const Vector3 AC = p3-p1;
    const Vector3 AQ = q -p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const SReal det = defaulttype::determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
    {
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
        SReal pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
        // barycentric coordinate on AC
        SReal pAC = b[1] / A[1][1]; // AQ*AB / AB*AB
        if (pAB < 0.000001 && pAC < 0.0000001)
        {
            // closest point is A
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1)) return 0; // this corner is not considered
            alpha = 0.0;
            beta = 0.0;
        }
        else if (pAB < 0.999999 && beta < 0.000001)
        {
            // closest point is on AB
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)) return 0; // this edge is not considered
            alpha = pAB;
            beta = 0.0;
        }
        else if (pAC < 0.999999 && alpha < 0.000001)
        {
            // closest point is on AC
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)) return 0; // this edge is not considered
            alpha = 0.0;
            beta = pAC;
        }
        else
        {
            // barycentric coordinate on BC
            // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
            SReal pBC = (b[1] - b[0] + A[0][0] - A[0][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
            if (pBC < 0.000001)
            {
                // closest point is B
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2)) return 0; // this edge is not considered
                alpha = 1.0;
                beta = 0.0;
            }
            else if (pBC > 0.999999)
            {
                // closest point is C
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3)) return 0; // this edge is not considered
                alpha = 0.0;
                beta = 1.0;
            }
            else
            {
                // closest point is on BC
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E23)) return 0; // this edge is not considered
                alpha = 1.0-pBC;
                beta = pBC;
            }
        }
    }

    Vector3 p, pq;
    p = p1 + AB * alpha + AC * beta;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq;
    }
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;

    ///!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value, you have to substract contactDist

    return 1;
}

int MeshIntToolUtil::projectPointOnTriangle(int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3, Vector3 & to_be_projected)
{
    const Vector3 AB = p2-p1;
    const Vector3 AC = p3-p1;
    const Vector3 AQ = to_be_projected -p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const SReal det = defaulttype::determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
    {
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
        SReal pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
        // barycentric coordinate on AC
        SReal pAC = b[1] / A[1][1]; // AQ*AB / AB*AB
        if (pAB < 0.000001 && pAC < 0.0000001)
        {
            // closest point is A
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1)) return 0; // this corner is not considered
            alpha = 0.0;
            beta = 0.0;
        }
        else if (pAB < 0.999999 && beta < 0.000001)
        {
            // closest point is on AB
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)) return 0; // this edge is not considered
            alpha = pAB;
            beta = 0.0;
        }
        else if (pAC < 0.999999 && alpha < 0.000001)
        {
            // closest point is on AC
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)) return 0; // this edge is not considered
            alpha = 0.0;
            beta = pAC;
        }
        else
        {
            // barycentric coordinate on BC
            // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
            SReal pBC = (b[1] - b[0] + A[0][0] - A[0][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
            if (pBC < 0.000001)
            {
                // closest point is B
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2)) return 0; // this edge is not considered
                alpha = 1.0;
                beta = 0.0;
            }
            else if (pBC > 0.999999)
            {
                // closest point is C
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3)) return 0; // this edge is not considered
                alpha = 0.0;
                beta = 1.0;
            }
            else
            {
                // closest point is on BC
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)) return 0; // this edge is not considered
                alpha = 1.0-pBC;
                beta = pBC;
            }
        }
    }

    to_be_projected = p1 + AB * alpha + AC * beta;

    return 1;
}

void MeshIntToolUtil::triangleBaryCoords(const Vector3& to_be_projected,const Vector3& p1, const Vector3& p2, const Vector3& p3,SReal & alpha,SReal & beta){
    const Vector3 AB = p2-p1;
    const Vector3 AC = p3-p1;
    const Vector3 AQ = to_be_projected -p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const SReal det = defaulttype::determinant(A);

    alpha = 0.5;
    beta = 0.5;

    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    if (alpha < 0 || beta < 0 || alpha + beta > 1)
    {
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
        SReal pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
        // barycentric coordinate on AC
        SReal pAC = b[1] / A[1][1]; // AQ*AC / AB*AB
        if (pAB < 0 && pAC < 0)
        {
            // closest point is A
            alpha = 0.0;
            beta = 0.0;
        }
        else if (pAB < 1 && beta < 0)
        {
            // closest point is on AB
            alpha = pAB;
            beta = 0.0;
        }
        else if (pAC < 1 && alpha < 0)
        {
            // closest point is on AC
            alpha = 0.0;
            beta = pAC;
        }
        else
        {
            // barycentric coordinate on BC
            // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
            SReal pBC = (b[1] - b[0] + A[0][0] - A[0][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
            if (pBC < 0)
            {
                // closest point is B
                alpha = 1.0;
                beta = 0.0;
            }
            else if (pBC > 1)
            {
                // closest point is C
                alpha = 0.0;
                beta = 1.0;
            }
            else
            {
                // closest point is on BC
                alpha = 1.0-pBC;
                beta = pBC;
            }
        }
    }
}



int MeshIntToolUtil::computeIntersection(Triangle& tri, int flags, OBB& obb, SReal alarmDist, SReal contactDist, OutputVector* contacts) {
    IntrTriangleOBB intr(tri, obb);
    if (intr.Find(alarmDist, flags)) {
        OBB::Real dist2 = (intr.pointOnFirst() - intr.pointOnSecond()).norm2();
        if ((!intr.colliding()) && dist2 > alarmDist * alarmDist)
            return 0;

        contacts->resize(contacts->size() + 1);
        DetectionOutput* detection = &*(contacts->end() - 1);

        detection->normal = intr.separatingAxis();
        detection->point[0] = intr.pointOnFirst();
        detection->point[1] = intr.pointOnSecond();

        if (intr.colliding())
            detection->value = -helper::rsqrt(dist2) - contactDist;
        else
            detection->value = helper::rsqrt(dist2) - contactDist;

        detection->elem.first = tri;
        detection->elem.second = obb;
        //detection->id = (tri.getCollisionModel()->getSize() > obb.getCollisionModel()->getSize()) ? tri.getIndex() : obb.getIndex();
        detection->id = tri.getIndex();

        return 1;
    }

    return 0;
}


template SOFA_SOFAMESHCOLLISION_API int MeshIntToolUtil::doCapPointInt(TCapsule<Vec3Types>& cap, const Vector3& q, SReal alarmDist, SReal contactDist, OutputVector* contacts);
template SOFA_SOFAMESHCOLLISION_API int MeshIntToolUtil::doCapLineInt(TCapsule<Vec3Types>& cap, const Vector3& q1, const Vector3& q2, SReal alarmDist, SReal contactDist, OutputVector* contacts, bool ignore_p1, bool ignore_p2);
template SOFA_SOFAMESHCOLLISION_API int MeshIntToolUtil::doCapPointInt(TCapsule<Rigid3Types>& cap, const Vector3& q, SReal alarmDist, SReal contactDist, OutputVector* contacts);
template SOFA_SOFAMESHCOLLISION_API int MeshIntToolUtil::doCapLineInt(TCapsule<Rigid3Types>& cap, const Vector3& q1, const Vector3& q2, SReal alarmDist, SReal contactDist, OutputVector* contacts, bool ignore_p1, bool ignore_p2);

template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Vec3Types>, Point>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Vec3Types>, Line>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Vec3Types>, Triangle>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Rigid3Types>, Point>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Rigid3Types>, Line>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TCapsule<sofa::defaulttype::Rigid3Types>, Triangle>;

template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TSphere<sofa::defaulttype::Vec3Types>, Point>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<Line, TSphere<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<Triangle, TSphere<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<TSphere<sofa::defaulttype::Rigid3Types>, Point>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<Line, TSphere<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BaseIntTool<Triangle, TSphere<sofa::defaulttype::Rigid3Types>>;

} // namespace sofa::component::collision
