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
#include <sofa/gl/CoreProfileRenderer.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/gl/shaders/drawToolGL.cppglsl>

#include <cmath>
#include <algorithm>
#include <cstring>

namespace sofa::gl
{

// Static member definitions
GLuint CoreProfileRenderer::s_vao = 0;
GLuint CoreProfileRenderer::s_vbo = 0;
GLsizeiptr CoreProfileRenderer::s_vboCapacity = 0;

GLuint CoreProfileRenderer::s_shaderProgram = 0;
bool CoreProfileRenderer::s_shaderReady = false;

GLint CoreProfileRenderer::s_locModelViewMatrix = -1;
GLint CoreProfileRenderer::s_locProjectionMatrix = -1;
GLint CoreProfileRenderer::s_locNormalMatrix = -1;
GLint CoreProfileRenderer::s_locLightingEnabled = -1;
GLint CoreProfileRenderer::s_locLightPosition = -1;
GLint CoreProfileRenderer::s_locLightAmbient = -1;
GLint CoreProfileRenderer::s_locLightDiffuse = -1;
GLint CoreProfileRenderer::s_locLightSpecular = -1;
GLint CoreProfileRenderer::s_locShininess = -1;

float CoreProfileRenderer::s_viewMatrix[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
float CoreProfileRenderer::s_projectionMatrix[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
float CoreProfileRenderer::s_lightPos[4]  = {0.0f, 0.0f, 1.0f, 0.0f};
float CoreProfileRenderer::s_lightAmb[4]  = {0.2f, 0.2f, 0.2f, 1.0f};
float CoreProfileRenderer::s_lightDif[4]  = {0.8f, 0.8f, 0.8f, 1.0f};
float CoreProfileRenderer::s_lightSpec[4] = {1.0f, 1.0f, 1.0f, 1.0f};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// State propagation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CoreProfileRenderer::setViewMatrix(const float mat[16])
{
    std::copy(mat, mat + 16, s_viewMatrix);
}

void CoreProfileRenderer::setProjectionMatrix(const float mat[16])
{
    std::copy(mat, mat + 16, s_projectionMatrix);
}

void CoreProfileRenderer::setLightPosition(float x, float y, float z, float w)
{
    s_lightPos[0] = x; s_lightPos[1] = y; s_lightPos[2] = z; s_lightPos[3] = w;
}

void CoreProfileRenderer::setLightAmbient(float r, float g, float b, float a)
{
    s_lightAmb[0] = r; s_lightAmb[1] = g; s_lightAmb[2] = b; s_lightAmb[3] = a;
}

void CoreProfileRenderer::setLightDiffuse(float r, float g, float b, float a)
{
    s_lightDif[0] = r; s_lightDif[1] = g; s_lightDif[2] = b; s_lightDif[3] = a;
}

void CoreProfileRenderer::setLightSpecular(float r, float g, float b, float a)
{
    s_lightSpec[0] = r; s_lightSpec[1] = g; s_lightSpec[2] = b; s_lightSpec[3] = a;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix math helpers
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CoreProfileRenderer::mat4Identity(float m[16])
{
    std::fill(m, m + 16, 0.0f);
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

void CoreProfileRenderer::mat4Multiply(float result[16], const float a[16], const float b[16])
{
    float tmp[16];
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
        {
            tmp[col * 4 + row] = 0.0f;
            for (int k = 0; k < 4; ++k)
                tmp[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
        }
    std::copy(tmp, tmp + 16, result);
}

void CoreProfileRenderer::computeNormalMatrix(const float mv[16], float nm[9])
{
    const float a00 = mv[0], a01 = mv[4], a02 = mv[8];
    const float a10 = mv[1], a11 = mv[5], a12 = mv[9];
    const float a20 = mv[2], a21 = mv[6], a22 = mv[10];

    const float det = a00 * (a11 * a22 - a12 * a21)
                    - a01 * (a10 * a22 - a12 * a20)
                    + a02 * (a10 * a21 - a11 * a20);

    if (std::fabs(det) < 1e-12f)
    {
        nm[0] = 1; nm[1] = 0; nm[2] = 0;
        nm[3] = 0; nm[4] = 1; nm[5] = 0;
        nm[6] = 0; nm[7] = 0; nm[8] = 1;
        return;
    }

    const float invDet = 1.0f / det;

    nm[0] = (a11 * a22 - a12 * a21) * invDet;
    nm[1] = (a12 * a20 - a10 * a22) * invDet;
    nm[2] = (a10 * a21 - a11 * a20) * invDet;
    nm[3] = (a02 * a21 - a01 * a22) * invDet;
    nm[4] = (a00 * a22 - a02 * a20) * invDet;
    nm[5] = (a20 * a01 - a00 * a21) * invDet;
    nm[6] = (a01 * a12 - a02 * a11) * invDet;
    nm[7] = (a02 * a10 - a00 * a12) * invDet;
    nm[8] = (a00 * a11 - a01 * a10) * invDet;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GL initialization
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static GLuint compileShaderStage(GLenum type, const std::string& source)
{
    const GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        GLint logLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 1)
        {
            std::string log(logLen, '\0');
            glGetShaderInfoLog(shader, logLen, nullptr, log.data());
            msg_error("CoreProfileRenderer") << "Shader compile error:\n" << log;
        }
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

void CoreProfileRenderer::init()
{
    if (s_shaderReady) return;

    const GLuint vs = compileShaderStage(GL_VERTEX_SHADER, drawToolGL_VertexShader);
    if (!vs) return;

    const GLuint fs = compileShaderStage(GL_FRAGMENT_SHADER, drawToolGL_FragmentShader);
    if (!fs) { glDeleteShader(vs); return; }

    s_shaderProgram = glCreateProgram();
    glAttachShader(s_shaderProgram, vs);
    glAttachShader(s_shaderProgram, fs);
    glLinkProgram(s_shaderProgram);

    glDetachShader(s_shaderProgram, vs);
    glDetachShader(s_shaderProgram, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = 0;
    glGetProgramiv(s_shaderProgram, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        GLint logLen = 0;
        glGetProgramiv(s_shaderProgram, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 1)
        {
            std::string log(logLen, '\0');
            glGetProgramInfoLog(s_shaderProgram, logLen, nullptr, log.data());
            msg_error("CoreProfileRenderer") << "Shader link error:\n" << log;
        }
        glDeleteProgram(s_shaderProgram);
        s_shaderProgram = 0;
        return;
    }

    s_locModelViewMatrix  = glGetUniformLocation(s_shaderProgram, "u_modelViewMatrix");
    s_locProjectionMatrix = glGetUniformLocation(s_shaderProgram, "u_projectionMatrix");
    s_locNormalMatrix     = glGetUniformLocation(s_shaderProgram, "u_normalMatrix");
    s_locLightingEnabled  = glGetUniformLocation(s_shaderProgram, "u_lightingEnabled");
    s_locLightPosition    = glGetUniformLocation(s_shaderProgram, "u_lightPosition");
    s_locLightAmbient     = glGetUniformLocation(s_shaderProgram, "u_lightAmbient");
    s_locLightDiffuse     = glGetUniformLocation(s_shaderProgram, "u_lightDiffuse");
    s_locLightSpecular    = glGetUniformLocation(s_shaderProgram, "u_lightSpecular");
    s_locShininess        = glGetUniformLocation(s_shaderProgram, "u_shininess");

    glGenVertexArrays(1, &s_vao);
    glBindVertexArray(s_vao);

    glGenBuffers(1, &s_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, s_vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, color)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    s_shaderReady = true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rendering
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CoreProfileRenderer::render(const std::vector<Vertex>& vertices, GLenum mode, bool lighting, const float modelMatrix[16])
{
    if (vertices.empty()) return;

    init();
    if (!s_shaderReady) return;

    // Compute modelview = view * model
    float modelView[16];
    if (modelMatrix)
        mat4Multiply(modelView, s_viewMatrix, modelMatrix);
    else
        std::copy(s_viewMatrix, s_viewMatrix + 16, modelView);

    float normalMatrix[9];
    computeNormalMatrix(modelView, normalMatrix);

    const GLsizeiptr requiredSize = static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex));

    glBindVertexArray(s_vao);
    glBindBuffer(GL_ARRAY_BUFFER, s_vbo);

    if (requiredSize > s_vboCapacity)
    {
        s_vboCapacity = requiredSize * 2;
        glBufferData(GL_ARRAY_BUFFER, s_vboCapacity, nullptr, GL_STREAM_DRAW);
    }
    else
    {
        glBufferData(GL_ARRAY_BUFFER, s_vboCapacity, nullptr, GL_STREAM_DRAW);
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0, requiredSize, vertices.data());

    glUseProgram(s_shaderProgram);

    glUniformMatrix4fv(s_locModelViewMatrix, 1, GL_FALSE, modelView);
    glUniformMatrix4fv(s_locProjectionMatrix, 1, GL_FALSE, s_projectionMatrix);
    glUniformMatrix3fv(s_locNormalMatrix, 1, GL_TRUE, normalMatrix);

    glUniform1i(s_locLightingEnabled, lighting ? 1 : 0);
    if (lighting)
    {
        glUniform4f(s_locLightPosition, s_lightPos[0], s_lightPos[1], s_lightPos[2], s_lightPos[3]);
        glUniform4f(s_locLightAmbient, s_lightAmb[0], s_lightAmb[1], s_lightAmb[2], s_lightAmb[3]);
        glUniform4f(s_locLightDiffuse, s_lightDif[0], s_lightDif[1], s_lightDif[2], s_lightDif[3]);
        glUniform4f(s_locLightSpecular, s_lightSpec[0], s_lightSpec[1], s_lightSpec[2], s_lightSpec[3]);
        glUniform1f(s_locShininess, 20.0f);
    }

    glDrawArrays(mode, 0, static_cast<GLsizei>(vertices.size()));

    glUseProgram(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void CoreProfileRenderer::renderTriangles(const std::vector<Vertex>& vertices, bool lighting, const float modelMatrix[16])
{
    render(vertices, GL_TRIANGLES, lighting, modelMatrix);
}

void CoreProfileRenderer::renderLines(const std::vector<Vertex>& vertices, const float modelMatrix[16])
{
    render(vertices, GL_LINES, false, modelMatrix);
}

void CoreProfileRenderer::renderWithModelView(const std::vector<Vertex>& vertices, GLenum mode, bool lighting, const float modelViewMatrix[16])
{
    if (vertices.empty()) return;

    init();
    if (!s_shaderReady) return;

    float normalMatrix[9];
    computeNormalMatrix(modelViewMatrix, normalMatrix);

    const GLsizeiptr requiredSize = static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex));

    glBindVertexArray(s_vao);
    glBindBuffer(GL_ARRAY_BUFFER, s_vbo);

    if (requiredSize > s_vboCapacity)
    {
        s_vboCapacity = requiredSize * 2;
        glBufferData(GL_ARRAY_BUFFER, s_vboCapacity, nullptr, GL_STREAM_DRAW);
    }
    else
    {
        glBufferData(GL_ARRAY_BUFFER, s_vboCapacity, nullptr, GL_STREAM_DRAW);
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0, requiredSize, vertices.data());

    glUseProgram(s_shaderProgram);

    glUniformMatrix4fv(s_locModelViewMatrix, 1, GL_FALSE, modelViewMatrix);
    glUniformMatrix4fv(s_locProjectionMatrix, 1, GL_FALSE, s_projectionMatrix);
    glUniformMatrix3fv(s_locNormalMatrix, 1, GL_TRUE, normalMatrix);

    glUniform1i(s_locLightingEnabled, lighting ? 1 : 0);
    if (lighting)
    {
        glUniform4f(s_locLightPosition, s_lightPos[0], s_lightPos[1], s_lightPos[2], s_lightPos[3]);
        glUniform4f(s_locLightAmbient, s_lightAmb[0], s_lightAmb[1], s_lightAmb[2], s_lightAmb[3]);
        glUniform4f(s_locLightDiffuse, s_lightDif[0], s_lightDif[1], s_lightDif[2], s_lightDif[3]);
        glUniform4f(s_locLightSpecular, s_lightSpec[0], s_lightSpec[1], s_lightSpec[2], s_lightSpec[3]);
        glUniform1f(s_locShininess, 20.0f);
    }

    glDrawArrays(mode, 0, static_cast<GLsizei>(vertices.size()));

    glUseProgram(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Geometry generators
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CoreProfileRenderer::generateSphereTriangles(std::vector<Vertex>& out,
    float cx, float cy, float cz,
    float rx, float ry, float rz,
    const float color[4], unsigned int rings, unsigned int sectors)
{
    const float R = 1.0f / static_cast<float>(rings - 1);
    const float S = 1.0f / static_cast<float>(sectors - 1);

    out.reserve(out.size() + rings * sectors * 6);

    for (unsigned int r = 0; r < rings - 1; ++r)
    {
        for (unsigned int s = 0; s < sectors - 1; ++s)
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
            const float x11 = std::cos(2.0f * static_cast<float>(M_PI) * (s + 1) * S) * yr1;
            const float z11 = std::sin(2.0f * static_cast<float>(M_PI) * (s + 1) * S) * yr1;

            auto pushVert = [&](float px, float py, float pz)
            {
                Vertex v;
                v.position[0] = cx + rx * px; v.position[1] = cy + ry * py; v.position[2] = cz + rz * pz;
                float nnx = px / rx, nny = py / ry, nnz = pz / rz;
                float nLen = std::sqrt(nnx*nnx + nny*nny + nnz*nnz);
                if (nLen > 0) { nnx /= nLen; nny /= nLen; nnz /= nLen; }
                v.normal[0] = nnx; v.normal[1] = nny; v.normal[2] = nnz;
                v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2]; v.color[3] = color[3];
                out.push_back(v);
            };

            pushVert(x00, y0, z00);
            pushVert(x10, y1, z10);
            pushVert(x11, y1, z11);

            pushVert(x00, y0, z00);
            pushVert(x11, y1, z11);
            pushVert(x01, y0, z01);
        }
    }
}

void CoreProfileRenderer::generateConeTriangles(std::vector<Vertex>& out,
    const float p1[3], const float p2[3],
    float radius1, float radius2,
    const float color[4], int subd)
{
    // Direction vector
    float dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];

    // Build perpendicular basis vectors
    float px = dx, py = dy, pz = dz;
    if (std::fabs(px) + std::fabs(py) < 0.00001f * std::sqrt(dx*dx + dy*dy + dz*dz))
        px += 1.0f;
    else
        pz += 1.0f;

    // q = p cross dir
    float qx = py * dz - pz * dy;
    float qy = pz * dx - px * dz;
    float qz = px * dy - py * dx;
    // p = dir cross q
    px = dy * qz - dz * qy;
    py = dz * qx - dx * qz;
    pz = dx * qy - dy * qx;

    // Normalize p
    float pLen = std::sqrt(px*px + py*py + pz*pz);
    if (pLen > 0) { px /= pLen; py /= pLen; pz /= pLen; }
    // Normalize q
    float qLen = std::sqrt(qx*qx + qy*qy + qz*qz);
    if (qLen > 0) { qx /= qLen; qy /= qLen; qz /= qLen; }

    // Direction for end caps
    float dirLen = std::sqrt(dx*dx + dy*dy + dz*dz);
    float ndx = 0, ndy = 0, ndz = 0;
    if (dirLen > 0) { ndx = dx/dirLen; ndy = dy/dirLen; ndz = dz/dirLen; }

    out.reserve(out.size() + subd * 12); // body + 2 caps, generous estimate

    // Generate side triangles (convert strip to individual triangles)
    for (int i = 0; i < subd; ++i)
    {
        const float theta0 = static_cast<float>(i * 2.0 * M_PI / subd);
        const float theta1 = static_cast<float>((i + 1) * 2.0 * M_PI / subd);
        const float ct0 = std::cos(theta0), st0 = std::sin(theta0);
        const float ct1 = std::cos(theta1), st1 = std::sin(theta1);

        // Normal directions
        float nx0 = px*ct0 + qx*st0, ny0 = py*ct0 + qy*st0, nz0 = pz*ct0 + qz*st0;
        float nx1 = px*ct1 + qx*st1, ny1 = py*ct1 + qy*st1, nz1 = pz*ct1 + qz*st1;

        // Points on disk 1 (at p1)
        float bx0 = p1[0] + nx0 * std::fabs(radius1);
        float by0 = p1[1] + ny0 * std::fabs(radius1);
        float bz0 = p1[2] + nz0 * std::fabs(radius1);
        float bx1 = p1[0] + nx1 * std::fabs(radius1);
        float by1 = p1[1] + ny1 * std::fabs(radius1);
        float bz1 = p1[2] + nz1 * std::fabs(radius1);

        // Points on disk 2 (at p2)
        float tx0 = p2[0] + nx0 * std::fabs(radius2);
        float ty0 = p2[1] + ny0 * std::fabs(radius2);
        float tz0 = p2[2] + nz0 * std::fabs(radius2);
        float tx1 = p2[0] + nx1 * std::fabs(radius2);
        float ty1 = p2[1] + ny1 * std::fabs(radius2);
        float tz1 = p2[2] + nz1 * std::fabs(radius2);

        auto pushV = [&](float vx, float vy, float vz, float vnx, float vny, float vnz)
        {
            Vertex v;
            v.position[0] = vx; v.position[1] = vy; v.position[2] = vz;
            v.normal[0] = vnx; v.normal[1] = vny; v.normal[2] = vnz;
            v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2]; v.color[3] = color[3];
            out.push_back(v);
        };

        // Two triangles for the quad between the two rings
        pushV(bx0, by0, bz0, nx0, ny0, nz0);
        pushV(tx0, ty0, tz0, nx0, ny0, nz0);
        pushV(tx1, ty1, tz1, nx1, ny1, nz1);

        pushV(bx0, by0, bz0, nx0, ny0, nz0);
        pushV(tx1, ty1, tz1, nx1, ny1, nz1);
        pushV(bx1, by1, bz1, nx1, ny1, nz1);

        // Cap at p1 (facing -direction)
        if (radius1 > 0)
        {
            pushV(p1[0], p1[1], p1[2], -ndx, -ndy, -ndz);
            pushV(bx1, by1, bz1, -ndx, -ndy, -ndz);
            pushV(bx0, by0, bz0, -ndx, -ndy, -ndz);
        }

        // Cap at p2 (facing +direction)
        if (radius2 > 0)
        {
            pushV(p2[0], p2[1], p2[2], ndx, ndy, ndz);
            pushV(tx0, ty0, tz0, ndx, ndy, ndz);
            pushV(tx1, ty1, tz1, ndx, ndy, ndz);
        }
    }
}

void CoreProfileRenderer::generateCylinderTriangles(std::vector<Vertex>& out,
    const float p1[3], const float p2[3],
    float radius, const float color[4], int subd)
{
    generateConeTriangles(out, p1, p2, radius, radius, color, subd);
}

void CoreProfileRenderer::generateArrowTriangles(std::vector<Vertex>& out,
    const float p1[3], const float p2[3],
    float radius, const float color[4], int subd)
{
    // p3 = p1*0.2 + p2*0.8  (80% of the way is cylinder, 20% is cone)
    float p3[3] = {
        p1[0] * 0.2f + p2[0] * 0.8f,
        p1[1] * 0.2f + p2[1] * 0.8f,
        p1[2] * 0.2f + p2[2] * 0.8f
    };
    generateCylinderTriangles(out, p1, p3, radius, color, subd);
    generateConeTriangles(out, p3, p2, radius * 2.5f, 0.0f, color, subd);
}

} // namespace sofa::gl
