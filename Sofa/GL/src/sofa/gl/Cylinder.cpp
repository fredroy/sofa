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
#include <sofa/gl/Cylinder.h>

#include <sofa/gl/CoreProfileRenderer.h>

#include <cassert>
#include <algorithm>
#include <iostream>
#include <cmath>


namespace sofa::gl
{

const int Cylinder::quadricDiscretisation = 16;

//GLuint Cylinder::displayList;
//GLUquadricObj *Cylinder::quadratic = nullptr;
std::map < std::pair<std::pair<float,float>,float>, Cylinder* > Cylinder::CylinderMap;

void Cylinder::initDraw()
{
    if (m_geometryReady) return;

    const float yellow[4] = {1.0f, 1.0f, 0.0f, 1.0f};

    // X axis cylinder
    if (length[0] > 0.0)
    {
        float rad = (float)(length[0] / 15.0);
        float p1[3] = {(float)(-length[0] / 2.0), 0, 0};
        float p2[3] = {(float)(length[0] / 2.0), 0, 0};
        CoreProfileRenderer::generateSphereTriangles(m_cachedVerts, p1[0], 0, 0, rad, rad, rad, yellow, quadricDiscretisation, quadricDiscretisation / 2);
        CoreProfileRenderer::generateCylinderTriangles(m_cachedVerts, p1, p2, rad, yellow, quadricDiscretisation);
        CoreProfileRenderer::generateSphereTriangles(m_cachedVerts, p2[0], 0, 0, rad, rad, rad, yellow, quadricDiscretisation, quadricDiscretisation / 2);
    }

    // Y axis cylinder
    if (length[1] > 0.0)
    {
        float rad = (float)(length[1] / 15.0);
        float p1[3] = {0, (float)(-length[1] / 2.0), 0};
        float p2[3] = {0, (float)(length[1] / 2.0), 0};
        CoreProfileRenderer::generateSphereTriangles(m_cachedVerts, 0, p1[1], 0, rad, rad, rad, yellow, quadricDiscretisation, quadricDiscretisation / 2);
        CoreProfileRenderer::generateCylinderTriangles(m_cachedVerts, p1, p2, rad, yellow, quadricDiscretisation);
        CoreProfileRenderer::generateSphereTriangles(m_cachedVerts, 0, p2[1], 0, rad, rad, rad, yellow, quadricDiscretisation, quadricDiscretisation / 2);
    }

    // Z axis cylinder
    if (length[2] > 0.0)
    {
        float rad = (float)(length[2] / 15.0);
        float p1[3] = {0, 0, (float)(-length[2] / 2.0)};
        float p2[3] = {0, 0, (float)(length[2] / 2.0)};
        CoreProfileRenderer::generateSphereTriangles(m_cachedVerts, 0, 0, p1[2], rad, rad, rad, yellow, quadricDiscretisation, quadricDiscretisation / 2);
        CoreProfileRenderer::generateCylinderTriangles(m_cachedVerts, p1, p2, rad, yellow, quadricDiscretisation);
        CoreProfileRenderer::generateSphereTriangles(m_cachedVerts, 0, 0, p2[2], rad, rad, rad, yellow, quadricDiscretisation, quadricDiscretisation / 2);
    }

    m_geometryReady = true;
}

void Cylinder::draw()
{
    initDraw();

    float modelMat[16];
    for (int i = 0; i < 16; ++i)
        modelMat[i] = static_cast<float>(matTransOpenGL[i]);

    CoreProfileRenderer::renderTriangles(m_cachedVerts, true, modelMat);
}

void Cylinder::update(const double *mat)
{
    std::copy(mat,mat+16, matTransOpenGL);
}

void Cylinder::update(const Vec3& center, const double orient[4][4])
{
    matTransOpenGL[0] = orient[0][0];
    matTransOpenGL[1] = orient[0][1];
    matTransOpenGL[2] = orient[0][2];
    matTransOpenGL[3] = 0;

    matTransOpenGL[4] = orient[1][0];
    matTransOpenGL[5] = orient[1][1];
    matTransOpenGL[6] = orient[1][2];
    matTransOpenGL[7] = 0;

    matTransOpenGL[8] = orient[2][0];
    matTransOpenGL[9] = orient[2][1];
    matTransOpenGL[10]= orient[2][2];
    matTransOpenGL[11] = 0;

    matTransOpenGL[12] = center[0];
    matTransOpenGL[13] = center[1];
    matTransOpenGL[14] = center[2];
    matTransOpenGL[15] = 1;
}

void Cylinder::update(const Vec3& center, const Quaternion& orient)
{
    orient.writeOpenGlMatrix(matTransOpenGL);
    matTransOpenGL[12] = center[0];
    matTransOpenGL[13] = center[1];
    matTransOpenGL[14] = center[2];
}

Cylinder::Cylinder(SReal len)
{
    length = Vec3(len,len,len);
    update(Vec3(0_sreal,0_sreal,0_sreal),  Quaternion(1_sreal,0_sreal,0_sreal,0_sreal));
}

Cylinder::Cylinder(const Vec3& len)
{
    length = len;
    update(Vec3(0_sreal,0_sreal,0_sreal),  Quaternion(1_sreal,0_sreal,0_sreal,0_sreal));
}

Cylinder::Cylinder(const Vec3& center, const Quaternion& orient, const Vec3& len)
{
    length = len;
    update(center, orient);
}

Cylinder::Cylinder(const Vec3& center, const double orient[4][4], const Vec3& len)
{
    length = len;
    update(center, orient);
}

Cylinder::Cylinder(const double *mat, const Vec3& len)
{
    length = len;
    update(mat);
}

Cylinder::Cylinder(const Vec3& center, const Quaternion& orient, SReal len)
{
    length = Vec3(len,len,len);
    update(center, orient);
}
Cylinder::Cylinder(const Vec3& center, const double orient[4][4], SReal len)
{
    length = Vec3(len,len,len);
    update(center, orient);
}

Cylinder::Cylinder(const double *mat, SReal len)
{
    length = Vec3(len,len,len);
    update(mat);
}

Cylinder::~Cylinder()
{
}

Cylinder* Cylinder::get(const Vec3& len)
{
    Cylinder*& a = CylinderMap[std::make_pair(std::make_pair((float)len[0],(float)len[1]),(float)len[2])];
    if (a==nullptr)
        a = new Cylinder(len);
    return a;
}

void Cylinder::draw(const Vec3& center, const Quaternion& orient, const Vec3& len)
{
    Cylinder* a = get(len);
    a->update(center, orient);
    a->draw();
}

void Cylinder::draw(const Vec3& center, const double orient[4][4], const Vec3& len)
{
    Cylinder* a = get(len);
    a->update(center, orient);
    a->draw();
}

void Cylinder::draw(const double* mat, const Vec3& len)
{
    Cylinder* a = get(len);
    a->update(mat);
    a->draw();
}

void Cylinder::draw(const Vec3& center, const Quaternion& orient, SReal len)
{
    Cylinder* a = get(Vec3(len, len, len));
    a->update(center, orient);
    a->draw();
}

void Cylinder::draw(const Vec3& center, const double orient[4][4], SReal len)
{
    Cylinder* a = get(Vec3(len, len, len));
    a->update(center, orient);
    a->draw();
}

void Cylinder::draw(const double* mat, SReal len)
{
    Cylinder* a = get(Vec3(len, len, len));
    a->update(mat);
    a->draw();
}

} // namespace sofa::gl
