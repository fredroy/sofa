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

#include <sofa/gl/config.h>
#include <sofa/gl/gl.h>
#include <vector>

namespace sofa::gl
{

class SOFA_GL_API CoreProfileRenderer
{
public:
    struct Vertex
    {
        float position[3];
        float normal[3];
        float color[4];
    }; // 40 bytes, same layout as DrawToolGL::DrawVertex

    // State propagation (called by DrawToolGL)
    static void setViewMatrix(const float mat[16]);
    static void setProjectionMatrix(const float mat[16]);
    static void setLightPosition(float x, float y, float z, float w);
    static void setLightAmbient(float r, float g, float b, float a);
    static void setLightDiffuse(float r, float g, float b, float a);
    static void setLightSpecular(float r, float g, float b, float a);

    // Rendering
    static void renderTriangles(const std::vector<Vertex>& vertices, bool lighting, const float modelMatrix[16] = nullptr);
    static void renderLines(const std::vector<Vertex>& vertices, const float modelMatrix[16] = nullptr);

    // Geometry generators (fill vectors, do not render)
    static void generateSphereTriangles(std::vector<Vertex>& out,
        float cx, float cy, float cz,
        float rx, float ry, float rz,
        const float color[4], unsigned int rings, unsigned int sectors);

    static void generateConeTriangles(std::vector<Vertex>& out,
        const float p1[3], const float p2[3],
        float radius1, float radius2,
        const float color[4], int subd = 16);

    static void generateCylinderTriangles(std::vector<Vertex>& out,
        const float p1[3], const float p2[3],
        float radius, const float color[4], int subd = 16);

    static void generateArrowTriangles(std::vector<Vertex>& out,
        const float p1[3], const float p2[3],
        float radius, const float color[4], int subd = 16);

    // Matrix math helpers
    static void mat4Identity(float m[16]);
    static void mat4Multiply(float result[16], const float a[16], const float b[16]);
    static void computeNormalMatrix(const float modelview[16], float normalMatrix3x3[9]);

private:
    static void init();
    static void render(const std::vector<Vertex>& vertices, GLenum mode, bool lighting, const float modelMatrix[16]);

    static GLuint s_vao;
    static GLuint s_vbo;
    static GLsizeiptr s_vboCapacity;

    static GLuint s_shaderProgram;
    static bool s_shaderReady;

    static GLint s_locModelViewMatrix;
    static GLint s_locProjectionMatrix;
    static GLint s_locNormalMatrix;
    static GLint s_locLightingEnabled;
    static GLint s_locLightPosition;
    static GLint s_locLightAmbient;
    static GLint s_locLightDiffuse;
    static GLint s_locLightSpecular;
    static GLint s_locShininess;

    static float s_viewMatrix[16];
    static float s_projectionMatrix[16];
    static float s_lightPos[4];
    static float s_lightAmb[4];
    static float s_lightDif[4];
    static float s_lightSpec[4];
};

} // namespace sofa::gl
