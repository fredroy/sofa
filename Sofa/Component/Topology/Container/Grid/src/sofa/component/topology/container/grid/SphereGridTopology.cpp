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
#include <sofa/component/topology/container/grid/SphereGridTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::grid
{

using namespace sofa::type;
using namespace sofa::defaulttype;



int SphereGridTopologyClass = core::RegisterObject("Sphere grid in 3D")
        .addAlias("SphereGrid")
        .add< SphereGridTopology >()
        ;

SphereGridTopology::SphereGridTopology(int nx, int ny, int nz)
    : GridTopology(nx, ny, nz)
    , d_center(initData(&d_center,Vector3(0_sreal, 0_sreal, 0_sreal),"center", "Center of the cylinder"))
    , d_axis(initData(&d_axis,Vector3(0_sreal , 0_sreal, 1_sreal),"axis", "Main direction of the cylinder"))
    , d_radius(initData(&d_radius, 1_sreal,"radius", "Radius of the cylinder"))
{
}

SphereGridTopology::SphereGridTopology()
    : GridTopology()
    , d_center(initData(&d_center,Vector3(0_sreal, 0_sreal, 0_sreal),"center", "Center of the cylinder"))
    , d_axis(initData(&d_axis,Vector3(0_sreal , 0_sreal, 1_sreal),"axis", "Main direction of the cylinder"))
    , d_radius(initData(&d_radius, 1_sreal,"radius", "Radius of the cylinder"))
{
}

void SphereGridTopology::setCenter(SReal x, SReal y, SReal z)
{
    d_center.emplaceValue(x,y,z);
}

void SphereGridTopology::setAxis(SReal x, SReal y, SReal z)
{
    d_axis.emplaceValue(x,y,z);
}

void SphereGridTopology::setRadius(SReal radius)
{
    d_radius.setValue(radius);
}


sofa::type::Vec3 SphereGridTopology::getPoint(Index i) const
{
    int x = i%d_n.getValue()[0]; i/=d_n.getValue()[0];
    int y = i%d_n.getValue()[1]; i/=d_n.getValue()[1];
    int z = i%d_n.getValue()[2]; i/=d_n.getValue()[2];
    return getPointInGrid(x,y,z);
}

sofa::type::Vec3 SphereGridTopology::getPointInGrid(int i, int j, int k) const
{
    //return p0+dx*x+dy*y+dz*z;
    SReal r = d_radius.getValue();
    Vector3 axisZ = d_axis.getValue();
    axisZ.normalize();
    Vector3 axisX = ((axisZ-Vector3(1_sreal,0_sreal,0_sreal)).norm() < 0.000001 ? Vector3(0_sreal,1_sreal,0_sreal) : Vector3(1_sreal,0_sreal,0_sreal));
    Vector3 axisY = cross(axisZ,axisX);
    axisX = cross(axisY,axisZ);
    axisX.normalize();
    axisY.normalize();
    axisZ.normalize();
    int nx = getNx();
    int ny = getNy();
    int nz = getNz();
    // coordonate on a square
    Vector3 p(i*2*r/(nx-1) - r, j*2*r/(ny-1) - r, k*2*r/(nz-1) - r);
    // scale it to be on a circle
    if (p.norm() > 0.0000001){
        SReal maxVal = helper::rmax(helper::rabs(p[0]),helper::rabs(p[1]));
        maxVal = helper::rmax(maxVal,helper::rabs(p[2]));
        p *= maxVal/p.norm();
    }

    return d_center.getValue()+axisX*p[0] + axisY*p[1] + axisZ * p[2];
}

} // namespace sofa::component::topology::container::grid
