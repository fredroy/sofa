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
#include <sofa/gl/Axis.h>

#include <sofa/gl/gl.h>
#include <sofa/gl/CoreProfileRenderer.h>

#include <cassert>
#include <algorithm>
#include <iostream>
#include <cmath>


namespace sofa::gl
{

const int Axis::quadricDiscretisation = 16;

//GLuint Axis::displayList;
//GLUquadricObj *Axis::quadratic = nullptr;
std::map < type::Vec3f, Axis::AxisSPtr > Axis::axisMap; // great idea but no more valid when creating a new opengl context when switching sofa viewer

void Axis::initDraw()
{
    if (m_geometryReady) return;

    type::Vec3 L = length;
    SReal Lmin = L[0];
    if (L[1] < Lmin) Lmin = L[1];
    if (L[2] < Lmin) Lmin = L[2];
    SReal Lmax = L[0];
    if (L[1] > Lmax) Lmax = L[1];
    if (L[2] > Lmax) Lmax = L[2];
    if (Lmax > Lmin * 2 && Lmin > 0.0)
        Lmax = Lmin * 2;
    if (Lmax > Lmin * 2)
        Lmin = Lmax / 1.414_sreal;

    type::Vec3 l(Lmin / 10_sreal, Lmin / 10_sreal, Lmin / 10_sreal);
    type::Vec3 lc(Lmax / 5_sreal, Lmax / 5_sreal, Lmax / 5_sreal);
    type::Vec3 Lc = lc;

    // White placeholder color - will be overwritten at draw time
    const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // X axis: center sphere + cylinder along X + cone arrowhead
    CoreProfileRenderer::generateSphereTriangles(m_xAxisVerts,
        0, 0, 0, (float)l[0], (float)l[0], (float)l[0], white, quadricDiscretisation, quadricDiscretisation / 2);

    if (L[0] > 0.0)
    {
        float p1[3] = {0, 0, 0};
        float p2[3] = {(float)L[0], 0, 0};
        CoreProfileRenderer::generateCylinderTriangles(m_xAxisVerts, p1, p2, (float)l[0], white, quadricDiscretisation);
        float p3[3] = {(float)L[0], 0, 0};
        float p4[3] = {(float)(L[0] + Lc[0]), 0, 0};
        CoreProfileRenderer::generateConeTriangles(m_xAxisVerts, p3, p4, (float)lc[0], 0.0f, white, quadricDiscretisation);
    }

    // Y axis: cylinder along Y + cone arrowhead
    if (L[1] > 0.0)
    {
        float p1[3] = {0, 0, 0};
        float p2[3] = {0, (float)L[1], 0};
        CoreProfileRenderer::generateCylinderTriangles(m_yAxisVerts, p1, p2, (float)l[1], white, quadricDiscretisation);
        float p3[3] = {0, (float)L[1], 0};
        float p4[3] = {0, (float)(L[1] + Lc[1]), 0};
        CoreProfileRenderer::generateConeTriangles(m_yAxisVerts, p3, p4, (float)lc[1], 0.0f, white, quadricDiscretisation);
    }

    // Z axis: cylinder along Z + cone arrowhead
    if (L[2] > 0.0)
    {
        float p1[3] = {0, 0, 0};
        float p2[3] = {0, 0, (float)L[2]};
        CoreProfileRenderer::generateCylinderTriangles(m_zAxisVerts, p1, p2, (float)l[2], white, quadricDiscretisation);
        float p3[3] = {0, 0, (float)L[2]};
        float p4[3] = {0, 0, (float)(L[2] + Lc[2])};
        CoreProfileRenderer::generateConeTriangles(m_zAxisVerts, p3, p4, (float)lc[2], 0.0f, white, quadricDiscretisation);
    }

    m_geometryReady = true;
}

void Axis::draw(const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    initDraw();

    float modelMat[16];
    for (int i = 0; i < 16; ++i)
        modelMat[i] = static_cast<float>(matTransOpenGL[i]);

    auto renderAxisWithColor = [&](std::vector<CoreProfileRenderer::Vertex>& verts, const type::RGBAColor& col)
    {
        if (verts.empty()) return;
        for (auto& v : verts)
        {
            v.color[0] = col[0]; v.color[1] = col[1]; v.color[2] = col[2]; v.color[3] = col[3];
        }
        CoreProfileRenderer::renderTriangles(verts, true, modelMat);
    };

    renderAxisWithColor(m_xAxisVerts, colorX);
    renderAxisWithColor(m_yAxisVerts, colorY);
    renderAxisWithColor(m_zAxisVerts, colorZ);
}

void Axis::update(const double *mat)
{
    std::copy(mat,mat+16, matTransOpenGL);
}

void Axis::update(const type::Vec3& center, const double orient[4][4])
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

void Axis::update(const type::Vec3& center, const Quaternion& orient)
{
    orient.writeOpenGlMatrix(matTransOpenGL);
    matTransOpenGL[12] = center[0];
    matTransOpenGL[13] = center[1];
    matTransOpenGL[14] = center[2];
}

Axis::Axis(SReal len)
{
    length = type::Vec3(len,len,len);
    update(type::Vec3(0_sreal,0_sreal,0_sreal),  Quaternion(1_sreal,0_sreal,0_sreal,0_sreal));
}

Axis::Axis(const type::Vec3& len)
{
    length = len;
    update(type::Vec3(0_sreal,0_sreal,0_sreal),  Quaternion(1_sreal,0_sreal,0_sreal,0_sreal));
}

Axis::Axis(const type::Vec3& center, const Quaternion& orient, const type::Vec3& len)
{
    length = len;
    update(center, orient);
}

Axis::Axis(const type::Vec3& center, const double orient[4][4], const type::Vec3& len)
{
    length = len;
    update(center, orient);
}

Axis::Axis(const double *mat, const type::Vec3& len)
{
    length = len;
    update(mat);
}

Axis::Axis(const type::Vec3& center, const Quaternion& orient, SReal len)
{
    length = type::Vec3(len,len,len);
    update(center, orient);
}
Axis::Axis(const type::Vec3& center, const double orient[4][4], SReal len)
{
    length = type::Vec3(len,len,len);
    update(center, orient);
}

Axis::Axis(const double *mat, SReal len)
{
    length = type::Vec3(len,len,len);
    update(mat);
}

Axis::~Axis()
{
}

Axis::AxisSPtr Axis::get(const type::Vec3& len)
{
    auto& a = axisMap[ { float(len[0]),float(len[1]),float(len[2]) }];
    if (a==nullptr)
        a = std::make_shared<Axis>(len);
    return a;
}

void Axis::draw(const type::Vec3& center, const Quaternion& orient, const type::Vec3& len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    const auto a = get(len);
    a->update(center, orient);
    a->draw(colorX, colorY, colorZ);
}

void Axis::draw(const type::Vec3& center, const double orient[4][4], const type::Vec3& len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    const auto a = get(len);
    a->update(center, orient);
    a->draw(colorX, colorY, colorZ);
}

void Axis::draw(const double* mat, const type::Vec3& len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    const auto a = get(len);
    a->update(mat);
    a->draw(colorX, colorY, colorZ);
}

void Axis::draw(const type::Vec3& center, const Quaternion& orient, SReal len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    const auto a = get(type::Vec3(len, len, len));
    a->update(center, orient);
    a->draw(colorX, colorY, colorZ);
}

void Axis::draw(const type::Vec3& center, const double orient[4][4], SReal len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    const auto a = get(type::Vec3(len, len, len));
    a->update(center, orient);
    a->draw(colorX, colorY, colorZ);
}

void Axis::draw(const double* mat, SReal len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    const auto a = get(type::Vec3(len, len, len));
    a->update(mat);
    a->draw(colorX, colorY, colorZ);
}

void Axis::draw(const type::Vec3& p1, const type::Vec3& p2, const double& r)
{
    const float fp1[3] = {(float)p1[0], (float)p1[1], (float)p1[2]};
    const float fp2[3] = {(float)p2[0], (float)p2[1], (float)p2[2]};
    const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<CoreProfileRenderer::Vertex> verts;
    CoreProfileRenderer::generateArrowTriangles(verts, fp1, fp2, (float)r, white, 16);
    CoreProfileRenderer::renderTriangles(verts, true);
}

void Axis::draw(const type::Vec3& p1, const type::Vec3& p2, const double& r1, const double& r2)
{
    const float fp1[3] = {(float)p1[0], (float)p1[1], (float)p1[2]};
    const float fp2[3] = {(float)p2[0], (float)p2[1], (float)p2[2]};
    const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<CoreProfileRenderer::Vertex> verts;
    CoreProfileRenderer::generateConeTriangles(verts, fp1, fp2, (float)r1, (float)r2, white, 16);
    CoreProfileRenderer::renderTriangles(verts, true);
}

} // namespace sofa::gl
