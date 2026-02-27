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
#include <sofa/gl/template.h>
#include <sofa/type/fixed_array.h>
#include <sofa/gl/CoreProfileRenderer.h>
#include <cmath>

namespace sofa::gl
{

template <typename V>
void drawCone(const V& p1, const V& p2, const float& radius1, const float& radius2, const int subd=8)
{
    const float fp1[3] = { static_cast<float>(p1[0]), static_cast<float>(p1[1]), static_cast<float>(p1[2]) };
    const float fp2[3] = { static_cast<float>(p2[0]), static_cast<float>(p2[1]), static_cast<float>(p2[2]) };
    const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    std::vector<CoreProfileRenderer::Vertex> verts;
    CoreProfileRenderer::generateConeTriangles(verts, fp1, fp2, radius1, radius2, white, subd);
    CoreProfileRenderer::renderTriangles(verts, true);
}

template <typename V>
void drawCylinder(const V& p1, const V& p2, const float& rad, const int subd=8)
{
    drawCone(p1, p2, rad, rad, subd);
}

template <typename V>
void drawArrow(const V& p1, const V& p2, const float& rad, const int subd=8)
{
    const float fp1[3] = { static_cast<float>(p1[0]), static_cast<float>(p1[1]), static_cast<float>(p1[2]) };
    const float fp2[3] = { static_cast<float>(p2[0]), static_cast<float>(p2[1]), static_cast<float>(p2[2]) };
    const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    std::vector<CoreProfileRenderer::Vertex> verts;
    CoreProfileRenderer::generateArrowTriangles(verts, fp1, fp2, rad, white, subd);
    CoreProfileRenderer::renderTriangles(verts, true);
}

template <typename V>
void drawSphere(const V& center, const float& rad, const int subd1=8, const int subd2=8)
{
    const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    std::vector<CoreProfileRenderer::Vertex> verts;
    CoreProfileRenderer::generateSphereTriangles(verts,
        static_cast<float>(center[0]), static_cast<float>(center[1]), static_cast<float>(center[2]),
        rad, rad, rad, white, subd1, subd2);
    CoreProfileRenderer::renderTriangles(verts, true);
}

template <typename V>
void drawEllipsoid(const V& center, const float& radx, const float& rady, const float& radz, const int subd1=8, const int subd2=8)
{
    const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    std::vector<CoreProfileRenderer::Vertex> verts;
    CoreProfileRenderer::generateSphereTriangles(verts,
        static_cast<float>(center[0]), static_cast<float>(center[1]), static_cast<float>(center[2]),
        radx, rady, radz, white, subd1, subd2);
    CoreProfileRenderer::renderTriangles(verts, true);
}

template <typename V>
void drawWireSphere(const V& center, const float& rad, const int subd1=8, const int subd2=8)
{
    const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    const float cx = static_cast<float>(center[0]);
    const float cy = static_cast<float>(center[1]);
    const float cz = static_cast<float>(center[2]);

    std::vector<CoreProfileRenderer::Vertex> verts;

    const float R = 1.0f / static_cast<float>(subd1 - 1);
    const float S = 1.0f / static_cast<float>(subd2 - 1);

    for (int r = 0; r < subd1 - 1; ++r)
    {
        for (int s = 0; s < subd2 - 1; ++s)
        {
            const float y0 = -std::cos(static_cast<float>(M_PI) * r * R);
            const float y1 = -std::cos(static_cast<float>(M_PI) * (r + 1) * R);
            const float yr0 = std::sin(static_cast<float>(M_PI) * r * R);
            const float yr1 = std::sin(static_cast<float>(M_PI) * (r + 1) * R);

            const float x00 = std::cos(2.0f * static_cast<float>(M_PI) * s * S) * yr0;
            const float z00 = std::sin(2.0f * static_cast<float>(M_PI) * s * S) * yr0;
            const float x10 = std::cos(2.0f * static_cast<float>(M_PI) * s * S) * yr1;
            const float z10 = std::sin(2.0f * static_cast<float>(M_PI) * s * S) * yr1;
            const float x01 = std::cos(2.0f * static_cast<float>(M_PI) * (s + 1) * S) * yr0;
            const float z01 = std::sin(2.0f * static_cast<float>(M_PI) * (s + 1) * S) * yr0;

            auto pushV = [&](float px, float py, float pz)
            {
                CoreProfileRenderer::Vertex v;
                v.position[0] = cx + rad * px; v.position[1] = cy + rad * py; v.position[2] = cz + rad * pz;
                v.normal[0] = px; v.normal[1] = py; v.normal[2] = pz;
                v.color[0] = white[0]; v.color[1] = white[1]; v.color[2] = white[2]; v.color[3] = white[3];
                verts.push_back(v);
            };

            // Horizontal line
            pushV(x00, y0, z00);
            pushV(x01, y0, z01);
            // Vertical line
            pushV(x00, y0, z00);
            pushV(x10, y1, z10);
        }
    }
    CoreProfileRenderer::renderLines(verts);
}

template <typename V>
void drawTorus(const float* coordinateMatrix, const float& bodyRad=0.0, const float& rad=1.0, const int precision=20,
               const V& color=sofa::type::fixed_array<int,3>({255,215,180}))
{
    const float c[4] = { color.x() / 255.0f, color.y() / 255.0f, color.z() / 255.0f, 1.0f };
    const float rr = 1.5f * bodyRad;
    const double dv = 2.0 * M_PI / precision;
    const double dw = 2.0 * M_PI / precision;

    std::vector<CoreProfileRenderer::Vertex> verts;

    double w = 0.0;
    while (w < 2.0 * M_PI + dw)
    {
        double v = 0.0;
        while (v < 2.0 * M_PI + dv)
        {
            auto pushV = [&](double vv, double ww)
            {
                CoreProfileRenderer::Vertex vert;
                vert.position[0] = static_cast<float>((rad + bodyRad * std::cos(vv)) * std::cos(ww));
                vert.position[1] = static_cast<float>((rad + bodyRad * std::cos(vv)) * std::sin(ww));
                vert.position[2] = static_cast<float>(bodyRad * std::sin(vv));
                float nx = static_cast<float>((rad + rr * std::cos(vv)) * std::cos(ww) - vert.position[0]);
                float ny = static_cast<float>((rad + rr * std::cos(vv)) * std::sin(ww) - vert.position[1]);
                float nz = static_cast<float>(rr * std::sin(vv) - bodyRad * std::sin(vv));
                float nLen = std::sqrt(nx*nx + ny*ny + nz*nz);
                if (nLen > 0) { nx /= nLen; ny /= nLen; nz /= nLen; }
                vert.normal[0] = nx; vert.normal[1] = ny; vert.normal[2] = nz;
                vert.color[0] = c[0]; vert.color[1] = c[1]; vert.color[2] = c[2]; vert.color[3] = c[3];
                verts.push_back(vert);
            };

            // Triangle 1
            pushV(v, w);
            pushV(v + dv, w + dw);
            pushV(v + dv, w);

            // Triangle 2
            pushV(v, w);
            pushV(v, w + dw);
            pushV(v + dv, w + dw);

            v += dv;
        }
        w += dw;
    }
    CoreProfileRenderer::renderTriangles(verts, true, coordinateMatrix);
}

template <typename V>
void drawEmptyParallelepiped(const V& vert1, const V& vert2, const V& vert3, const V& vert4, const V& vecFromFaceToOppositeFace, const float& rad=1.0, const int precision=8,
                             const V& color = sofa::type::RGBAColor::red())
{
    // Vertices of the parallelepiped
    drawSphere(vert1, rad);
    drawSphere(vert2, rad);
    drawSphere(vert3, rad);
    drawSphere(vert4, rad);
    drawSphere(vert1 + vecFromFaceToOppositeFace, rad);
    drawSphere(vert2 + vecFromFaceToOppositeFace, rad);
    drawSphere(vert3 + vecFromFaceToOppositeFace, rad);
    drawSphere(vert4 + vecFromFaceToOppositeFace, rad);

    // First face
    drawCylinder(vert1, vert2, rad, precision);
    drawCylinder(vert2, vert3, rad, precision);
    drawCylinder(vert3, vert4, rad, precision);
    drawCylinder(vert4, vert1, rad, precision);

    // The opposite face
    drawCylinder(vert1 + vecFromFaceToOppositeFace, vert2 + vecFromFaceToOppositeFace, rad, precision);
    drawCylinder(vert2 + vecFromFaceToOppositeFace, vert3 + vecFromFaceToOppositeFace, rad, precision);
    drawCylinder(vert3 + vecFromFaceToOppositeFace, vert4 + vecFromFaceToOppositeFace, rad, precision);
    drawCylinder(vert4 + vecFromFaceToOppositeFace, vert1 + vecFromFaceToOppositeFace, rad, precision);

    // Connect the two faces
    drawCylinder(vert1, vert1 + vecFromFaceToOppositeFace, rad, precision);
    drawCylinder(vert2, vert2 + vecFromFaceToOppositeFace, rad, precision);
    drawCylinder(vert3, vert3 + vecFromFaceToOppositeFace, rad, precision);
    drawCylinder(vert4, vert4 + vecFromFaceToOppositeFace, rad, precision);
}

} // namespace sofa::gl
