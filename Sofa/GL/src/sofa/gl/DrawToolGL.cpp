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
#define SOFA_HELPER_GL_DRAWTOOLGL_CPP

#include <sofa/gl/DrawToolGL.h>
#include <sofa/gl/CoreProfileRenderer.h>

#include <sofa/gl/gl.h>
#include <sofa/gl/Axis.h>
#include <sofa/gl/Cylinder.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/type/Mat.h>
#include <cmath>
#include <map>

#include <sofa/gl/shaders/drawToolGL.cppglsl>

namespace sofa::gl
{

using namespace sofa::type;
using sofa::type::RGBAColor;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix math helpers
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::mat4Identity(float* m)
{
    std::fill(m, m + 16, 0.0f);
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

void DrawToolGL::mat4Multiply(float* result, const float* a, const float* b)
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

void DrawToolGL::mat4Translation(float* m, float x, float y, float z)
{
    mat4Identity(m);
    m[12] = x; m[13] = y; m[14] = z;
}

void DrawToolGL::mat4Scale(float* m, float sx, float sy, float sz)
{
    std::fill(m, m + 16, 0.0f);
    m[0] = sx; m[5] = sy; m[10] = sz; m[15] = 1.0f;
}

const DrawToolGL::Mat16f& DrawToolGL::currentModelView() const
{
    return m_modelViewStack.top();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Modern GL infrastructure
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DrawToolGL::DrawToolGL()
{
    clear();
    mLightEnabled = false;
    mWireFrameEnabled = false;
    mPolygonMode = 1;

    // Initialize matrix stacks
    Mat16f identity;
    mat4Identity(identity.data());
    m_projectionMatrix = identity;
    m_modelViewStack.push(identity);
}

DrawToolGL::~DrawToolGL()
{
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
    if (m_shaderProgram) glDeleteProgram(m_shaderProgram);
}

void DrawToolGL::init()
{
}

void DrawToolGL::setProjectionMatrix(const double* mat16)
{
    for (int i = 0; i < 16; ++i)
        m_projectionMatrix[i] = static_cast<float>(mat16[i]);
    CoreProfileRenderer::setProjectionMatrix(m_projectionMatrix.data());
}

void DrawToolGL::setModelViewMatrix(const double* mat16)
{
    // Replace the current top of the modelview stack
    Mat16f mv;
    for (int i = 0; i < 16; ++i)
        mv[i] = static_cast<float>(mat16[i]);

    // Clear the stack and push the new base
    while (m_modelViewStack.size() > 1)
        m_modelViewStack.pop();
    m_modelViewStack.top() = mv;
    CoreProfileRenderer::setViewMatrix(mv.data());
}

void DrawToolGL::setLightPosition(float x, float y, float z, float w)
{
    m_lightPos[0] = x; m_lightPos[1] = y; m_lightPos[2] = z; m_lightPos[3] = w;
    CoreProfileRenderer::setLightPosition(x, y, z, w);
}

void DrawToolGL::setLightAmbient(float r, float g, float b, float a)
{
    m_lightAmb[0] = r; m_lightAmb[1] = g; m_lightAmb[2] = b; m_lightAmb[3] = a;
    CoreProfileRenderer::setLightAmbient(r, g, b, a);
}

void DrawToolGL::setLightDiffuse(float r, float g, float b, float a)
{
    m_lightDif[0] = r; m_lightDif[1] = g; m_lightDif[2] = b; m_lightDif[3] = a;
    CoreProfileRenderer::setLightDiffuse(r, g, b, a);
}

void DrawToolGL::setLightSpecular(float r, float g, float b, float a)
{
    m_lightSpec[0] = r; m_lightSpec[1] = g; m_lightSpec[2] = b; m_lightSpec[3] = a;
    CoreProfileRenderer::setLightSpecular(r, g, b, a);
}

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
            msg_error("DrawToolGL") << "Shader compile error:\n" << log;
        }
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

void DrawToolGL::initModernGL()
{
    if (m_shaderReady) return;

    // Compile vertex shader
    const GLuint vs = compileShaderStage(GL_VERTEX_SHADER, drawToolGL_VertexShader);
    if (!vs) return;

    // Compile fragment shader
    const GLuint fs = compileShaderStage(GL_FRAGMENT_SHADER, drawToolGL_FragmentShader);
    if (!fs) { glDeleteShader(vs); return; }

    // Link program
    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vs);
    glAttachShader(m_shaderProgram, fs);
    glLinkProgram(m_shaderProgram);

    // Shaders can be detached and deleted after linking
    glDetachShader(m_shaderProgram, vs);
    glDetachShader(m_shaderProgram, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = 0;
    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        GLint logLen = 0;
        glGetProgramiv(m_shaderProgram, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 1)
        {
            std::string log(logLen, '\0');
            glGetProgramInfoLog(m_shaderProgram, logLen, nullptr, log.data());
            msg_error("DrawToolGL") << "Shader link error:\n" << log;
        }
        glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
        return;
    }

    // Cache uniform locations
    m_locModelViewMatrix  = glGetUniformLocation(m_shaderProgram, "u_modelViewMatrix");
    m_locProjectionMatrix = glGetUniformLocation(m_shaderProgram, "u_projectionMatrix");
    m_locNormalMatrix     = glGetUniformLocation(m_shaderProgram, "u_normalMatrix");
    m_locLightingEnabled  = glGetUniformLocation(m_shaderProgram, "u_lightingEnabled");
    m_locLightPosition    = glGetUniformLocation(m_shaderProgram, "u_lightPosition");
    m_locLightAmbient     = glGetUniformLocation(m_shaderProgram, "u_lightAmbient");
    m_locLightDiffuse     = glGetUniformLocation(m_shaderProgram, "u_lightDiffuse");
    m_locLightSpecular    = glGetUniformLocation(m_shaderProgram, "u_lightSpecular");
    m_locShininess        = glGetUniformLocation(m_shaderProgram, "u_shininess");

    // Create VAO
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    // Create VBO
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    // Setup vertex attribute layout for DrawVertex (40 bytes)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DrawVertex),
                          reinterpret_cast<void*>(offsetof(DrawVertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(DrawVertex),
                          reinterpret_cast<void*>(offsetof(DrawVertex, normal)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(DrawVertex),
                          reinterpret_cast<void*>(offsetof(DrawVertex, color)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    m_shaderReady = true;
}

void DrawToolGL::computeNormalMatrix(const float* mv, float* nm)
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

void DrawToolGL::uploadMatrices()
{
    const auto& mv = currentModelView();
    float normalMatrix[9];
    computeNormalMatrix(mv.data(), normalMatrix);

    glUniformMatrix4fv(m_locModelViewMatrix, 1, GL_FALSE, mv.data());
    glUniformMatrix4fv(m_locProjectionMatrix, 1, GL_FALSE, m_projectionMatrix.data());
    glUniformMatrix3fv(m_locNormalMatrix, 1, GL_TRUE, normalMatrix);
}

void DrawToolGL::uploadLightState()
{
    glUniform4f(m_locLightPosition, m_lightPos[0], m_lightPos[1], m_lightPos[2], m_lightPos[3]);
    glUniform4f(m_locLightAmbient, m_lightAmb[0], m_lightAmb[1], m_lightAmb[2], m_lightAmb[3]);
    glUniform4f(m_locLightDiffuse, m_lightDif[0], m_lightDif[1], m_lightDif[2], m_lightDif[3]);
    glUniform4f(m_locLightSpecular, m_lightSpec[0], m_lightSpec[1], m_lightSpec[2], m_lightSpec[3]);
}

void DrawToolGL::flushVertexBuffer(GLenum mode, bool lighting)
{
    if (m_vertexBuffer.empty()) return;

    initModernGL();
    if (!m_shaderReady) return;

    const GLsizeiptr requiredSize = static_cast<GLsizeiptr>(m_vertexBuffer.size() * sizeof(DrawVertex));

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    if (requiredSize > m_vboCapacity)
    {
        m_vboCapacity = requiredSize * 2;
        glBufferData(GL_ARRAY_BUFFER, m_vboCapacity, nullptr, GL_STREAM_DRAW);
    }
    else
    {
        glBufferData(GL_ARRAY_BUFFER, m_vboCapacity, nullptr, GL_STREAM_DRAW);
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0, requiredSize, m_vertexBuffer.data());

    glUseProgram(m_shaderProgram);

    uploadMatrices();

    glUniform1i(m_locLightingEnabled, lighting ? 1 : 0);
    if (lighting)
    {
        uploadLightState();
        glUniform1f(m_locShininess, 20.0f);
    }

    glDrawArrays(mode, 0, static_cast<GLsizei>(m_vertexBuffer.size()));

    glUseProgram(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    m_vertexBuffer.clear();
}

void DrawToolGL::pushVertex(const Vec3& pos, const Vec3& normal, const RGBAColor& color)
{
    DrawVertex v;
    v.position[0] = static_cast<float>(pos[0]);
    v.position[1] = static_cast<float>(pos[1]);
    v.position[2] = static_cast<float>(pos[2]);
    v.normal[0] = static_cast<float>(normal[0]);
    v.normal[1] = static_cast<float>(normal[1]);
    v.normal[2] = static_cast<float>(normal[2]);
    v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2]; v.color[3] = color[3];
    m_vertexBuffer.push_back(v);
}

void DrawToolGL::pushVertex(float px, float py, float pz, float nx, float ny, float nz, const RGBAColor& color)
{
    DrawVertex v;
    v.position[0] = px; v.position[1] = py; v.position[2] = pz;
    v.normal[0] = nx; v.normal[1] = ny; v.normal[2] = nz;
    v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2]; v.color[3] = color[3];
    m_vertexBuffer.push_back(v);
}

void DrawToolGL::pushQuadAsTriangles(const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4,
        const Vec3& normal, const RGBAColor& color)
{
    pushVertex(p1, normal, color);
    pushVertex(p2, normal, color);
    pushVertex(p3, normal, color);
    pushVertex(p1, normal, color);
    pushVertex(p3, normal, color);
    pushVertex(p4, normal, color);
}

void DrawToolGL::pushQuadAsTriangles(const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4,
        const Vec3& normal,
        const RGBAColor& c1, const RGBAColor& c2, const RGBAColor& c3, const RGBAColor& c4)
{
    pushVertex(p1, normal, c1);
    pushVertex(p2, normal, c2);
    pushVertex(p3, normal, c3);
    pushVertex(p1, normal, c1);
    pushVertex(p3, normal, c3);
    pushVertex(p4, normal, c4);
}

void DrawToolGL::pushQuadAsTriangles(const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4,
        const Vec3& n1, const Vec3& n2, const Vec3& n3, const Vec3& n4,
        const RGBAColor& c1, const RGBAColor& c2, const RGBAColor& c3, const RGBAColor& c4)
{
    pushVertex(p1, n1, c1);
    pushVertex(p2, n2, c2);
    pushVertex(p3, n3, c3);
    pushVertex(p1, n1, c1);
    pushVertex(p3, n3, c3);
    pushVertex(p4, n4, c4);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Sphere generation (procedural UV sphere)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::generateSphereTriangles(const Vec3& center, float rx, float ry, float rz,
                                          const RGBAColor& color, unsigned int rings, unsigned int sectors)
{
    const float R = 1.0f / (float)(rings - 1);
    const float S = 1.0f / (float)(sectors - 1);

    m_vertexBuffer.reserve(m_vertexBuffer.size() + rings * sectors * 6);

    for (unsigned int r = 0; r < rings - 1; ++r)
    {
        for (unsigned int s = 0; s < sectors - 1; ++s)
        {
            // Four corners of the quad on the sphere
            float y0 = -std::cos(M_PI * r * R);
            float y1 = -std::cos(M_PI * (r + 1) * R);
            float yr0 = std::sin(M_PI * r * R);
            float yr1 = std::sin(M_PI * (r + 1) * R);

            float x00 = std::cos(2.0f * M_PI * s * S) * yr0;
            float z00 = std::sin(2.0f * M_PI * s * S) * yr0;
            float x10 = std::cos(2.0f * M_PI * s * S) * yr1;
            float z10 = std::sin(2.0f * M_PI * s * S) * yr1;
            float x01 = std::cos(2.0f * M_PI * (s + 1) * S) * yr0;
            float z01 = std::sin(2.0f * M_PI * (s + 1) * S) * yr0;
            float x11 = std::cos(2.0f * M_PI * (s + 1) * S) * yr1;
            float z11 = std::sin(2.0f * M_PI * (s + 1) * S) * yr1;

            float cx = static_cast<float>(center[0]);
            float cy = static_cast<float>(center[1]);
            float cz = static_cast<float>(center[2]);

            // Two triangles per quad
            pushVertex(cx + rx * x00, cy + ry * y0, cz + rz * z00, x00, y0, z00, color);
            pushVertex(cx + rx * x10, cy + ry * y1, cz + rz * z10, x10, y1, z10, color);
            pushVertex(cx + rx * x11, cy + ry * y1, cz + rz * z11, x11, y1, z11, color);

            pushVertex(cx + rx * x00, cy + ry * y0, cz + rz * z00, x00, y0, z00, color);
            pushVertex(cx + rx * x11, cy + ry * y1, cz + rz * z11, x11, y1, z11, color);
            pushVertex(cx + rx * x01, cy + ry * y0, cz + rz * z01, x01, y0, z01, color);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Point methods
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPoint(const Vec3 &p, const RGBAColor &c)
{
    const Vec3 dummyNormal(0, 0, 1);
    pushVertex(p, dummyNormal, c);
    flushVertexBuffer(GL_POINTS, false);
}

void DrawToolGL::drawPoint(const Vec3 &p, const Vec3 &n, const RGBAColor &c)
{
    pushVertex(p, n, c);
    flushVertexBuffer(GL_POINTS, false);
}

void DrawToolGL::drawPoints(const std::vector<Vec3> &points, float size, const RGBAColor& color)
{
    glPointSize(size);
    const Vec3 dummyNormal(0, 0, 1);
    m_vertexBuffer.reserve(points.size());
    for (const auto& p : points)
        pushVertex(p, dummyNormal, color);
    flushVertexBuffer(GL_POINTS, false);
    glPointSize(1);
}

void DrawToolGL::drawPoints(const std::vector<Vec3> &points, float size, const std::vector<RGBAColor>& color)
{
    glPointSize(size);
    const Vec3 dummyNormal(0, 0, 1);
    m_vertexBuffer.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
        pushVertex(points[i], dummyNormal, color[i]);
    flushVertexBuffer(GL_POINTS, false);
    glPointSize(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Line methods
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawLine(const Vec3 &p1, const Vec3 &p2, const RGBAColor& color)
{
    const Vec3 dummyNormal(0, 0, 1);
    pushVertex(p1, dummyNormal, color);
    pushVertex(p2, dummyNormal, color);
    flushVertexBuffer(GL_LINES, false);
}

void DrawToolGL::drawInfiniteLine(const Vec3 &point, const Vec3 &direction, const RGBAColor& color, const bool& vanishing)
{
    const Vec3 dummyNormal(0, 0, 1);
    pushVertex(point, dummyNormal, color);
    const Vec3 farPoint = point + direction * 100000.0;
    RGBAColor endColor = color;
    if (vanishing) endColor[3] = 0.0f;
    pushVertex(farPoint, dummyNormal, endColor);
    flushVertexBuffer(GL_LINES, false);
}

void DrawToolGL::drawInfiniteLine(const Vec3 &point, const Vec3 &direction, const float& size, const RGBAColor& color, const bool &vanishing)
{
    glLineWidth(size);
    drawInfiniteLine(point, direction, color, vanishing);
    glLineWidth(1);
}

void DrawToolGL::drawLines(const std::vector<Vec3> &points, float size, const RGBAColor& color)
{
    glLineWidth(size);
    const Vec3 dummyNormal(0, 0, 1);
    m_vertexBuffer.reserve(points.size());
    for (std::size_t i = 0; i < points.size() / 2; ++i)
    {
        pushVertex(points[2*i], dummyNormal, color);
        pushVertex(points[2*i+1], dummyNormal, color);
    }
    flushVertexBuffer(GL_LINES, false);
    glLineWidth(1);
}

void DrawToolGL::drawLines(const std::vector<Vec3> &points, float size, const std::vector<RGBAColor>& colors)
{
    if (points.size() != colors.size()*2)
    {
        msg_warning("DrawToolGL") << "Sizes mismatch in drawLines method";
        return drawLines(points, size, RGBAColor::red());
    }

    std::map<RGBAColor, std::vector<Vec3> > colorPointsMap;
    for (std::size_t i = 0; i < colors.size(); ++i)
    {
        colorPointsMap[colors[i]].push_back(points[2 * i]);
        colorPointsMap[colors[i]].push_back(points[2 * i + 1]);
    }

    for (const auto& [color, pts] : colorPointsMap)
        drawLines(pts, size, color);
}

void DrawToolGL::drawLines(const std::vector<Vec3> &points, const std::vector< Vec<2,int> > &index, float size, const RGBAColor& color)
{
    glLineWidth(size);
    const Vec3 dummyNormal(0, 0, 1);
    m_vertexBuffer.reserve(index.size() * 2);
    for (std::size_t i = 0; i < index.size(); ++i)
    {
        pushVertex(points[index[i][0]], dummyNormal, color);
        pushVertex(points[index[i][1]], dummyNormal, color);
    }
    flushVertexBuffer(GL_LINES, false);
    glLineWidth(1);
}

void DrawToolGL::drawLineStrip(const std::vector<Vec3> &points, float size, const RGBAColor& color)
{
    glLineWidth(size);
    const Vec3 dummyNormal(0, 0, 1);
    m_vertexBuffer.reserve(points.size());
    for (const auto& p : points)
        pushVertex(p, dummyNormal, color);
    flushVertexBuffer(GL_LINE_STRIP, false);
    glLineWidth(1);
}

void DrawToolGL::drawLineLoop(const std::vector<Vec3> &points, float size, const RGBAColor& color)
{
    glLineWidth(size);
    const Vec3 dummyNormal(0, 0, 1);
    m_vertexBuffer.reserve(points.size());
    for (const auto& p : points)
        pushVertex(p, dummyNormal, color);
    flushVertexBuffer(GL_LINE_LOOP, false);
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Disk and Circle
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawDisk(float radius, double from, double to, int resolution, const RGBAColor& color)
{
    if (from > to)
        to += 2.0 * M_PI;

    bool first = true;
    float prev_alpha = 0;
    float prev_beta = 0;

    for (int i = 0; i <= resolution; ++i)
    {
        double angle = (double(i) / double(resolution) * 2.0 * M_PI) + from;
        bool stop = false;
        if (angle >= to) { angle = to; stop = true; }

        const float alpha = float(std::sin(angle));
        const float beta = float(std::cos(angle));

        if (first)
        {
            first = false;
            prev_alpha = alpha;
            prev_beta = beta;
            if (stop) break;
            continue;
        }

        pushVertex(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, color);
        pushVertex(radius * prev_alpha, radius * prev_beta, 0.0f, 0.0f, 0.0f, 1.0f, color);
        pushVertex(radius * alpha, radius * beta, 0.0f, 0.0f, 0.0f, 1.0f, color);

        if (stop) break;
        prev_alpha = alpha;
        prev_beta = beta;
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawCircle(float radius, float lineThickness, int resolution, const RGBAColor& color)
{
    glLineWidth(lineThickness);
    m_vertexBuffer.reserve(resolution + 1);
    for (int i = 0; i <= resolution; ++i)
    {
        const float angle = float(double(i) / double(resolution) * 2.0 * M_PI);
        const float alpha = std::sin(angle);
        const float beta = std::cos(angle);
        pushVertex(radius * alpha, radius * beta, 0.0f, 0.0f, 0.0f, 1.0f, color);
    }
    flushVertexBuffer(GL_LINE_STRIP, false);
    glLineWidth(1.0f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Triangle methods
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangle(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const Vec3 &normal)
{
    const RGBAColor white = RGBAColor::white();
    pushVertex(p1, normal, white); pushVertex(p2, normal, white); pushVertex(p3, normal, white);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTriangle(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const Vec3 &normal, const RGBAColor &c)
{
    pushVertex(p1, normal, c); pushVertex(p2, normal, c); pushVertex(p3, normal, c);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTriangle(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3,
        const Vec3 &normal,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3)
{
    pushVertex(p1, normal, c1); pushVertex(p2, normal, c2); pushVertex(p3, normal, c3);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTriangle(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3,
        const Vec3 &normal1, const Vec3 &normal2, const Vec3 &normal3,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3)
{
    pushVertex(p1, normal1, c1); pushVertex(p2, normal2, c2); pushVertex(p3, normal3, c3);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTriangles(const std::vector<Vec3> &points, const RGBAColor& color)
{
    m_vertexBuffer.reserve(points.size());
    for (std::size_t i = 0; i < points.size() / 3; ++i)
    {
        const Vec3& a = points[3*i+0]; const Vec3& b = points[3*i+1]; const Vec3& c = points[3*i+2];
        Vec3 n = cross((b-a), (c-a)); n.normalize();
        pushVertex(a, n, color); pushVertex(b, n, color); pushVertex(c, n, color);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTriangles(const std::vector<Vec3> &points, const std::vector<RGBAColor> &color)
{
    std::vector<Vec3> normal;
    this->drawTriangles(points, normal, color);
}

void DrawToolGL::drawTriangles(const std::vector<Vec3> &points, const Vec3& normal, const RGBAColor& color)
{
    m_vertexBuffer.reserve(points.size());
    for (std::size_t i = 0; i < points.size() / 3; ++i)
    {
        pushVertex(points[3*i+0], normal, color);
        pushVertex(points[3*i+1], normal, color);
        pushVertex(points[3*i+2], normal, color);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTriangles(const std::vector<Vec3> &points,
        const std::vector< Vec<3,int> > &index,
        const std::vector<Vec3> &normal, const RGBAColor& color)
{
    m_vertexBuffer.reserve(index.size() * 3);
    for (std::size_t i = 0; i < index.size(); ++i)
    {
        pushVertex(points[index[i][0]], normal[i], color);
        pushVertex(points[index[i][1]], normal[i], color);
        pushVertex(points[index[i][2]], normal[i], color);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTriangles(const std::vector<Vec3> &points,
    const std::vector< Vec3i > &index,
    const std::vector<Vec3> &normal,
    const std::vector<RGBAColor>& colors)
{
    m_vertexBuffer.reserve(index.size() * 3);
    for (std::size_t i = 0; i < index.size(); ++i)
    {
        pushVertex(points[index[i][0]], normal[i], colors[3*i+0]);
        pushVertex(points[index[i][1]], normal[i], colors[3*i+1]);
        pushVertex(points[index[i][2]], normal[i], colors[3*i+2]);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTriangles(const std::vector<Vec3> &points,
        const std::vector<Vec3> &normal, const std::vector<RGBAColor> &color)
{
    const std::size_t nbTriangles = points.size() / 3;
    const bool computeNormals = (normal.size() != nbTriangles);
    if (nbTriangles == 0) return;
    m_vertexBuffer.reserve(nbTriangles * 3);
    for (std::size_t i = 0; i < nbTriangles; ++i)
    {
        if (!computeNormals)
        {
            pushVertex(points[3*i+0], normal[i], color[3*i+0]);
            pushVertex(points[3*i+1], normal[i], color[3*i+1]);
            pushVertex(points[3*i+2], normal[i], color[3*i+2]);
        }
        else
        {
            const Vec3& a = points[3*i+0]; const Vec3& b = points[3*i+1]; const Vec3& c = points[3*i+2];
            Vec3 n = cross((b-a), (c-a)); n.normalize();
            pushVertex(a, n, color[3*i+0]); pushVertex(b, n, color[3*i+1]); pushVertex(c, n, color[3*i+2]);
        }
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTriangleStrip(const std::vector<Vec3> &points,
        const std::vector<Vec3> &normal, const RGBAColor& color)
{
    m_vertexBuffer.reserve(normal.size() * 2);
    for (std::size_t i = 0; i < normal.size(); ++i)
    {
        pushVertex(points[2*i], normal[i], color);
        pushVertex(points[2*i+1], normal[i], color);
    }
    flushVertexBuffer(GL_TRIANGLE_STRIP);
}

void DrawToolGL::drawTriangleFan(const std::vector<Vec3> &points,
        const std::vector<Vec3> &normal, const RGBAColor& color)
{
    if (points.size() < 3) return;
    m_vertexBuffer.reserve(points.size());
    pushVertex(points[0], normal[0], color);
    pushVertex(points[1], normal[0], color);
    pushVertex(points[2], normal[0], color);
    for (std::size_t i = 3; i < points.size(); ++i)
        pushVertex(points[i], normal[i], color);
    flushVertexBuffer(GL_TRIANGLE_FAN);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Frame (reimplemented using modern draw methods)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawFrame(const Vec3& position, const Quaternion &orientation, const Vec<3,float> &size)
{
    // Draw colored axes: X=red, Y=green, Z=blue
    type::Matrix3 rotMat;
    orientation.toMatrix(rotMat);

    Vec3 xAxis(rotMat[0][0], rotMat[1][0], rotMat[2][0]);
    Vec3 yAxis(rotMat[0][1], rotMat[1][1], rotMat[2][1]);
    Vec3 zAxis(rotMat[0][2], rotMat[1][2], rotMat[2][2]);

    drawLine(position, position + xAxis * size[0], RGBAColor::red());
    drawLine(position, position + yAxis * size[1], RGBAColor::green());
    drawLine(position, position + zAxis * size[2], RGBAColor::blue());
}

void DrawToolGL::drawFrame(const Vec3& position, const Quaternion &orientation, const Vec<3,float> &size, const RGBAColor &color)
{
    type::Matrix3 rotMat;
    orientation.toMatrix(rotMat);

    Vec3 xAxis(rotMat[0][0], rotMat[1][0], rotMat[2][0]);
    Vec3 yAxis(rotMat[0][1], rotMat[1][1], rotMat[2][1]);
    Vec3 zAxis(rotMat[0][2], rotMat[1][2], rotMat[2][2]);

    drawLine(position, position + xAxis * size[0], color);
    drawLine(position, position + yAxis * size[1], color);
    drawLine(position, position + zAxis * size[2], color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Spheres (reimplemented using procedural mesh)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawSphere(const Vec3 &p, float radius)
{
    generateSphereTriangles(p, radius, radius, radius, RGBAColor::white(), 16, 16);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawSphere(const Vec3 &p, float radius, const RGBAColor &color)
{
    generateSphereTriangles(p, radius, radius, radius, color, 16, 16);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawEllipsoid(const Vec3 &p, const Vec3 &radii)
{
    generateSphereTriangles(p, (float)radii[0], (float)radii[1], (float)radii[2], RGBAColor::white(), 16, 16);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawSpheres(const std::vector<Vec3> &points, float radius, const RGBAColor& color)
{
    for (const auto& p : points)
        generateSphereTriangles(p, radius, radius, radius, color, 12, 12);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawSpheres(const std::vector<Vec3> &points, const std::vector<float>& radius, const RGBAColor& color)
{
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        float r = (i < radius.size()) ? radius[i] : radius[0];
        generateSphereTriangles(points[i], r, r, r, color, 12, 12);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawFakeSpheres(const std::vector<Vec3> &points, float radius, const RGBAColor& color)
{
    drawSpheres(points, radius, color);
}

void DrawToolGL::drawFakeSpheres(const std::vector<Vec3> &points, const std::vector<float>& radius, const RGBAColor& color)
{
    drawSpheres(points, radius, color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Capsule, Cone, Cube, Cylinder, Arrow, Cross, Plus
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawCapsule(const Vec3& p1, const Vec3 &p2, float radius, const RGBAColor& color, int subd)
{
    Vec3 tmp = p2-p1;
    Vec3 p=tmp;
    if (fabs(p[0]) + fabs(p[1]) < 0.00001*tmp.norm())
        p[0] += 1.0;
    else
        p[2] += 1.0;
    Vec3 q = p.cross(tmp);
    p = tmp.cross(q);
    p.normalize();
    q.normalize();

    std::vector<Vec3> points;
    std::vector<Vec3> normals;

    for (int i2=0 ; i2<=subd ; i2++)
    {
        const float theta = (float)( i2 * 2.0f * M_PI / subd );
        tmp = p*std::cos(theta)+q*std::sin(theta);
        normals.push_back(tmp);

        Vec3 w(p1);
        w += tmp*fabs(radius);
        points.push_back(w);
        w=p2;
        w += tmp*fabs(radius);
        points.push_back(w);
    }

    drawTriangleStrip(points, normals, color);
    drawSphere(p1, radius, color);
    drawSphere(p2, radius, color);
}

void DrawToolGL::drawCone(const Vec3& p1, const Vec3 &p2, float radius1, float radius2, const RGBAColor& color, int subd)
{
    Vec3 tmp = p2-p1;
    Vec3 p=tmp;
    if (fabs(p[0]) + fabs(p[1]) < 0.00001*tmp.norm())
        p[0] += 1.0;
    else
        p[2] += 1.0;
    Vec3 q = p.cross(tmp);
    p = tmp.cross(q);
    p.normalize();
    q.normalize();

    std::vector<Vec3> points;
    std::vector<Vec3> normals;

    std::vector<Vec3> pointsCloseCylinder1;
    std::vector<Vec3> normalsCloseCylinder1;
    std::vector<Vec3> pointsCloseCylinder2;
    std::vector<Vec3> normalsCloseCylinder2;

    Vec3 dir=p1-p2; dir.normalize();

    pointsCloseCylinder1.push_back(p1);
    normalsCloseCylinder1.push_back(dir);
    pointsCloseCylinder2.push_back(p2);
    normalsCloseCylinder2.push_back(-dir);

    for (int i2=0 ; i2<=subd ; i2++)
    {
        const float theta = (float)( i2 * 2.0f * M_PI / subd );
        tmp = p*std::cos(theta)+q*std::sin(theta);
        normals.push_back(tmp);

        Vec3 w(p1);
        w += tmp*fabs(radius1);
        points.push_back(w);
        pointsCloseCylinder1.push_back(w);
        normalsCloseCylinder1.push_back(dir);

        w=p2;
        w += tmp*fabs(radius2);
        points.push_back(w);
        pointsCloseCylinder2.push_back(w);
        normalsCloseCylinder2.push_back(-dir);
    }
    pointsCloseCylinder1.push_back(pointsCloseCylinder1[1]);
    normalsCloseCylinder1.push_back(normalsCloseCylinder1[1]);
    pointsCloseCylinder2.push_back(pointsCloseCylinder2[1]);
    normalsCloseCylinder2.push_back(normalsCloseCylinder2[1]);

    drawTriangleStrip(points, normals, color);
    if (radius1 > 0) drawTriangleFan(pointsCloseCylinder1, normalsCloseCylinder1, color);
    if (radius2 > 0) drawTriangleFan(pointsCloseCylinder2, normalsCloseCylinder2, color);
}

void DrawToolGL::drawCube(const float& radius, const RGBAColor& color, const int& subd)
{
    drawCylinder(Vec3(-1,-1,-1), Vec3(1,-1,-1), radius, color, subd);
    drawCylinder(Vec3(-1, 1,-1), Vec3(1, 1,-1), radius, color, subd);
    drawCylinder(Vec3(-1,-1, 1), Vec3(1,-1, 1), radius, color, subd);
    drawCylinder(Vec3(-1, 1, 1), Vec3(1, 1, 1), radius, color, subd);
    drawCylinder(Vec3(-1,-1,-1), Vec3(-1,1,-1), radius, color, subd);
    drawCylinder(Vec3(-1,-1, 1), Vec3(-1,1, 1), radius, color, subd);
    drawCylinder(Vec3( 1,-1,-1), Vec3( 1,1,-1), radius, color, subd);
    drawCylinder(Vec3( 1,-1, 1), Vec3( 1,1, 1), radius, color, subd);
    drawCylinder(Vec3(-1,-1,-1), Vec3(-1,-1,1), radius, color, subd);
    drawCylinder(Vec3(-1, 1,-1), Vec3(-1, 1,1), radius, color, subd);
    drawCylinder(Vec3( 1,-1,-1), Vec3( 1,-1,1), radius, color, subd);
    drawCylinder(Vec3( 1, 1,-1), Vec3( 1, 1,1), radius, color, subd);
}

void DrawToolGL::drawCylinder(const Vec3& p1, const Vec3 &p2, float radius, const RGBAColor& color, int subd)
{
    drawCone(p1, p2, radius, radius, color, subd);
}

void DrawToolGL::drawArrow(const Vec3& p1, const Vec3 &p2, float radius, const RGBAColor& color, int subd)
{
    const Vec3 p3 = p1*.2+p2*.8_sreal;
    drawCylinder(p1, p3, radius, color, subd);
    drawCone(p3, p2, radius*2.5f, 0, color, subd);
}

void DrawToolGL::drawArrow(const Vec3& p1, const Vec3 &p2, float radius, float coneLength, const RGBAColor& color, int subd)
{
    drawArrow(p1, p2, radius, coneLength, radius * 2.5f, color, subd);
}

void DrawToolGL::drawArrow(const Vec3& p1, const Vec3 &p2, float radius, float coneLength, float coneRadius, const RGBAColor& color, int subd)
{
    Vec3 a = p2 - p1;
    const SReal n = a.norm();
    if (coneLength >= n)
        drawCone(p1, p2, coneRadius, 0, color, subd);
    else
    {
        a /= n;
        const Vec3 p3 = p2 - coneLength*a;
        drawCylinder(p1, p3, radius, color, subd);
        drawCone(p3, p2, coneRadius, 0, color, subd);
    }
}

void DrawToolGL::drawCross(const Vec3&p, float length, const RGBAColor& color)
{
    std::vector<Vec3> bounds;
    for (unsigned int i=0; i<3; i++)
    {
        Vec3 p0 = p; Vec3 p1 = p;
        p0[i] -= length; p1[i] += length;
        bounds.push_back(p0); bounds.push_back(p1);
    }
    drawLines(bounds, 1, color);
}

void DrawToolGL::drawPlus(const float& radius, const RGBAColor& color, const int& subd)
{
    drawCylinder(Vec3(-1, 0, 0), Vec3(1, 0, 0), radius, color, subd);
    drawCylinder(Vec3(0, -1, 0), Vec3(0, 1, 0), radius, color, subd);
    drawCylinder(Vec3(0, 0, -1), Vec3(0, 0, 1), radius, color, subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Quad methods (GL_QUADS elimination)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawQuad(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const Vec3 &p4, const Vec3 &normal)
{
    pushQuadAsTriangles(p1, p2, p3, p4, normal, RGBAColor::white());
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawQuad(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const Vec3 &p4,
        const Vec3 &normal, const RGBAColor &c)
{
    pushQuadAsTriangles(p1, p2, p3, p4, normal, c);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawQuad(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const Vec3 &p4,
        const Vec3 &normal,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4)
{
    pushQuadAsTriangles(p1, p2, p3, p4, normal, c1, c2, c3, c4);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawQuad(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const Vec3 &p4,
        const Vec3 &normal1, const Vec3 &normal2, const Vec3 &normal3, const Vec3 &normal4,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4)
{
    pushQuadAsTriangles(p1, p2, p3, p4, normal1, normal2, normal3, normal4, c1, c2, c3, c4);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawQuads(const std::vector<Vec3> &points, const RGBAColor& color)
{
    m_vertexBuffer.reserve((points.size() / 4) * 6);
    for (std::size_t i = 0; i < points.size() / 4; ++i)
    {
        const Vec3& a = points[4*i+0]; const Vec3& b = points[4*i+1];
        const Vec3& c = points[4*i+2]; const Vec3& d = points[4*i+3];
        Vec3 n = cross((b-a), (c-a)); n.normalize();
        pushQuadAsTriangles(a, b, c, d, n, color);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawQuads(const std::vector<Vec3> &points, const std::vector<RGBAColor>& colors)
{
    m_vertexBuffer.reserve((points.size() / 4) * 6);
    for (std::size_t i = 0; i < points.size() / 4; ++i)
    {
        const Vec3& a = points[4*i+0]; const Vec3& b = points[4*i+1];
        const Vec3& c = points[4*i+2]; const Vec3& d = points[4*i+3];
        RGBAColor avg;
        for(int jj=0; jj<4; jj++)
            avg[jj] = (colors[4*i][jj]+colors[4*i+1][jj]+colors[4*i+2][jj]+colors[4*i+3][jj])*0.25f;
        Vec3 n = cross((b-a), (c-a)); n.normalize();
        pushQuadAsTriangles(a, b, c, d, n, avg);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tetrahedron methods
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTetrahedron(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const RGBAColor &color)
{
    Vec3 n;
    n = cross((p1-p0), (p2-p0)); n.normalize();
    pushVertex(p0, n, color); pushVertex(p1, n, color); pushVertex(p2, n, color);
    n = cross((p1-p0), (p3-p0)); n.normalize();
    pushVertex(p0, n, color); pushVertex(p1, n, color); pushVertex(p3, n, color);
    n = cross((p2-p0), (p3-p0)); n.normalize();
    pushVertex(p0, n, color); pushVertex(p2, n, color); pushVertex(p3, n, color);
    n = cross((p2-p1), (p3-p1)); n.normalize();
    pushVertex(p1, n, color); pushVertex(p2, n, color); pushVertex(p3, n, color);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawScaledTetrahedron(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, const RGBAColor& color, const float scale)
{
    const Vec3 center = (p0 + p1 + p2 + p3) / 4.0;
    const Vec3 np0 = ((p0-center)*scale)+center;
    const Vec3 np1 = ((p1-center)*scale)+center;
    const Vec3 np2 = ((p2-center)*scale)+center;
    const Vec3 np3 = ((p3-center)*scale)+center;

    Vec3 n;
    n = cross((p1-p0), (p2-p0)); n.normalize();
    pushVertex(np0, n, color); pushVertex(np1, n, color); pushVertex(np2, n, color);
    n = cross((p1-p0), (p3-p0)); n.normalize();
    pushVertex(np0, n, color); pushVertex(np1, n, color); pushVertex(np3, n, color);
    n = cross((p2-p0), (p3-p0)); n.normalize();
    pushVertex(np0, n, color); pushVertex(np2, n, color); pushVertex(np3, n, color);
    n = cross((p2-p1), (p3-p1)); n.normalize();
    pushVertex(np1, n, color); pushVertex(np2, n, color); pushVertex(np3, n, color);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawTetrahedra(const std::vector<Vec3> &points, const RGBAColor &color)
{
    m_vertexBuffer.reserve((points.size() / 4) * 12);
    for (auto it = points.begin(), end = points.end(); it != end;)
    {
        const Vec3& p0 = *(it++); const Vec3& p1 = *(it++);
        const Vec3& p2 = *(it++); const Vec3& p3 = *(it++);
        Vec3 n;
        n = cross((p1-p0),(p2-p0)); n.normalize();
        pushVertex(p0,n,color); pushVertex(p1,n,color); pushVertex(p2,n,color);
        n = cross((p1-p0),(p3-p0)); n.normalize();
        pushVertex(p0,n,color); pushVertex(p1,n,color); pushVertex(p3,n,color);
        n = cross((p2-p0),(p3-p0)); n.normalize();
        pushVertex(p0,n,color); pushVertex(p2,n,color); pushVertex(p3,n,color);
        n = cross((p2-p1),(p3-p1)); n.normalize();
        pushVertex(p1,n,color); pushVertex(p2,n,color); pushVertex(p3,n,color);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawScaledTetrahedra(const std::vector<Vec3> &points, const RGBAColor &color, const float scale)
{
    m_vertexBuffer.reserve((points.size() / 4) * 12);
    for (auto it = points.begin(), end = points.end(); it != end;)
    {
        const Vec3& p0 = *(it++); const Vec3& p1 = *(it++);
        const Vec3& p2 = *(it++); const Vec3& p3 = *(it++);
        Vec3 center = (p0+p1+p2+p3)/4.0;
        Vec3 np0 = ((p0-center)*scale)+center;
        Vec3 np1 = ((p1-center)*scale)+center;
        Vec3 np2 = ((p2-center)*scale)+center;
        Vec3 np3 = ((p3-center)*scale)+center;
        Vec3 n;
        n = cross((p1-p0),(p2-p0)); n.normalize();
        pushVertex(np0,n,color); pushVertex(np1,n,color); pushVertex(np2,n,color);
        n = cross((p1-p0),(p3-p0)); n.normalize();
        pushVertex(np0,n,color); pushVertex(np1,n,color); pushVertex(np3,n,color);
        n = cross((p2-p0),(p3-p0)); n.normalize();
        pushVertex(np0,n,color); pushVertex(np2,n,color); pushVertex(np3,n,color);
        n = cross((p2-p1),(p3-p1)); n.normalize();
        pushVertex(np1,n,color); pushVertex(np2,n,color); pushVertex(np3,n,color);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Hexahedron methods
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawHexahedron(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, const Vec3 &p3,
                                const Vec3 &p4, const Vec3 &p5, const Vec3 &p6, const Vec3 &p7,
                                const RGBAColor &color)
{
    Vec3 n;
    n = cross((p1-p0),(p2-p0)); n.normalize(); pushQuadAsTriangles(p0,p1,p2,p3,n,color);
    n = cross((p7-p5),(p6-p5)); n.normalize(); pushQuadAsTriangles(p4,p7,p6,p5,n,color);
    n = cross((p0-p1),(p4-p1)); n.normalize(); pushQuadAsTriangles(p1,p0,p4,p5,n,color);
    n = cross((p5-p1),(p6-p1)); n.normalize(); pushQuadAsTriangles(p1,p5,p6,p2,n,color);
    n = cross((p6-p2),(p7-p2)); n.normalize(); pushQuadAsTriangles(p2,p6,p7,p3,n,color);
    n = cross((p3-p0),(p7-p0)); n.normalize(); pushQuadAsTriangles(p0,p3,p7,p4,n,color);
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawHexahedra(const std::vector<Vec3> &points, const RGBAColor& color)
{
    m_vertexBuffer.reserve((points.size() / 8) * 36);
    for (auto it = points.begin(), end = points.end(); it != end;)
    {
        const Vec3& p0=*(it++); const Vec3& p1=*(it++); const Vec3& p2=*(it++); const Vec3& p3=*(it++);
        const Vec3& p4=*(it++); const Vec3& p5=*(it++); const Vec3& p6=*(it++); const Vec3& p7=*(it++);
        Vec3 n;
        n=cross((p1-p0),(p2-p0)); n.normalize(); pushQuadAsTriangles(p0,p1,p2,p3,n,color);
        n=cross((p7-p5),(p6-p5)); n.normalize(); pushQuadAsTriangles(p4,p7,p6,p5,n,color);
        n=cross((p0-p1),(p4-p1)); n.normalize(); pushQuadAsTriangles(p1,p0,p4,p5,n,color);
        n=cross((p5-p1),(p6-p1)); n.normalize(); pushQuadAsTriangles(p1,p5,p6,p2,n,color);
        n=cross((p6-p2),(p7-p2)); n.normalize(); pushQuadAsTriangles(p2,p6,p7,p3,n,color);
        n=cross((p3-p0),(p7-p0)); n.normalize(); pushQuadAsTriangles(p0,p3,p7,p4,n,color);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

void DrawToolGL::drawScaledHexahedra(const std::vector<Vec3> &points, const RGBAColor& color, const float scale)
{
    m_vertexBuffer.reserve((points.size() / 8) * 36);
    for (auto it = points.begin(), end = points.end(); it != end;)
    {
        const Vec3& p0=*(it++); const Vec3& p1=*(it++); const Vec3& p2=*(it++); const Vec3& p3=*(it++);
        const Vec3& p4=*(it++); const Vec3& p5=*(it++); const Vec3& p6=*(it++); const Vec3& p7=*(it++);
        Vec3 center=(p0+p1+p2+p3+p4+p5+p6+p7)/8.0;
        Vec3 np0=((p0-center)*scale)+center; Vec3 np1=((p1-center)*scale)+center;
        Vec3 np2=((p2-center)*scale)+center; Vec3 np3=((p3-center)*scale)+center;
        Vec3 np4=((p4-center)*scale)+center; Vec3 np5=((p5-center)*scale)+center;
        Vec3 np6=((p6-center)*scale)+center; Vec3 np7=((p7-center)*scale)+center;
        Vec3 n;
        n=cross((p1-p0),(p2-p0)); n.normalize(); pushQuadAsTriangles(np0,np1,np2,np3,n,color);
        n=cross((p7-p5),(p6-p5)); n.normalize(); pushQuadAsTriangles(np4,np7,np6,np5,n,color);
        n=cross((p0-p1),(p4-p1)); n.normalize(); pushQuadAsTriangles(np1,np0,np4,np5,n,color);
        n=cross((p5-p1),(p6-p1)); n.normalize(); pushQuadAsTriangles(np1,np5,np6,np2,n,color);
        n=cross((p6-p2),(p7-p2)); n.normalize(); pushQuadAsTriangles(np2,np6,np7,np3,n,color);
        n=cross((p3-p0),(p7-p0)); n.normalize(); pushQuadAsTriangles(np0,np3,np7,np4,n,color);
    }
    flushVertexBuffer(GL_TRIANGLES);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bounding box
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawBoundingBox(const Vec3 &min, const Vec3 &max, float size)
{
    glLineWidth(size);
    const Vec3 dummyNormal(0, 0, 1);
    const RGBAColor white = RGBAColor::white();
    m_vertexBuffer.reserve(24);

    pushVertex(Vec3(min[0],min[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],min[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],max[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],max[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],min[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],min[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],max[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],max[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],min[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],max[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],min[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],max[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],min[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],max[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],min[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],max[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],min[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],min[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],min[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],min[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],max[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(max[0],max[1],max[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],max[1],min[2]),dummyNormal,white);
    pushVertex(Vec3(min[0],max[1],max[2]),dummyNormal,white);

    flushVertexBuffer(GL_LINES, false);
    glLineWidth(1.0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// State and material methods (core profile compatible)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::setPolygonMode(int _mode, bool _wireframe)
{
    mPolygonMode = _mode;
    mWireFrameEnabled = _wireframe;
    // Core profile only supports GL_FRONT_AND_BACK
    if (mWireFrameEnabled)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void DrawToolGL::setLightingEnabled(bool _isAnabled)
{
    mLightEnabled = _isAnabled;
}

void DrawToolGL::setMaterial(const RGBAColor &color)
{
    // In core profile, material properties are handled via our shader uniforms.
    // We just manage blending state here.
    if (color[3] < 1)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(0);
    }
    else
    {
        glDisable(GL_BLEND);
        glDepthMask(1);
    }
}

void DrawToolGL::resetMaterial(const RGBAColor &color)
{
    if (color[3] < 1)
        resetMaterial();
}

void DrawToolGL::resetMaterial()
{
    glDisable(GL_BLEND);
    glDepthMask(1);
}

void DrawToolGL::clear()
{
    gl::Axis::clear();
    gl::Cylinder::clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix methods (internal stack, no legacy GL)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::pushMatrix()
{
    m_modelViewStack.push(m_modelViewStack.top());
}

void DrawToolGL::popMatrix()
{
    if (m_modelViewStack.size() > 1)
        m_modelViewStack.pop();
}

void DrawToolGL::multMatrix(float* glTransform)
{
    Mat16f result;
    mat4Multiply(result.data(), m_modelViewStack.top().data(), glTransform);
    m_modelViewStack.top() = result;
}

void DrawToolGL::scale(float s)
{
    float scaleMat[16];
    mat4Scale(scaleMat, s, s, s);
    Mat16f result;
    mat4Multiply(result.data(), m_modelViewStack.top().data(), scaleMat);
    m_modelViewStack.top() = result;
}

void DrawToolGL::translate(float x, float y, float z)
{
    float transMat[16];
    mat4Translation(transMat, x, y, z);
    Mat16f result;
    mat4Multiply(result.data(), m_modelViewStack.top().data(), transMat);
    m_modelViewStack.top() = result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Text (no-op in core profile - GlText uses fixed-function pipeline)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::writeOverlayText(int /*x*/, int /*y*/, unsigned /*fontSize*/, const RGBAColor& /*color*/, const char* /*text*/)
{
    // Text rendering requires fixed-function pipeline (GlText).
    // Not available in core profile. Intentionally left as no-op.
}

void DrawToolGL::draw3DText(const Vec3& /*p*/, float /*scale*/, const RGBAColor& /*color*/, const char* /*text*/)
{
    // Not available in core profile.
}

void DrawToolGL::draw3DText_Indices(const std::vector<Vec3>& /*positions*/, float /*scale*/, const RGBAColor& /*color*/)
{
    // Not available in core profile.
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// State methods (core profile compatible)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::enableBlending()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void DrawToolGL::disableBlending()
{
    glDisable(GL_BLEND);
}

void DrawToolGL::enablePolygonOffset(float factor, float units)
{
    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(factor, units);
}

void DrawToolGL::disablePolygonOffset()
{
    glDisable(GL_POLYGON_OFFSET_LINE);
}

void DrawToolGL::enableLighting()
{
    mLightEnabled = true;
}

void DrawToolGL::disableLighting()
{
    mLightEnabled = false;
}

void DrawToolGL::enableDepthTest()
{
    glEnable(GL_DEPTH_TEST);
}

void DrawToolGL::disableDepthTest()
{
    glDisable(GL_DEPTH_TEST);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Save/restore state (core profile compatible - manual save/restore)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::saveLastState()
{
    glGetBooleanv(GL_BLEND, &m_savedState.blend);
    glGetBooleanv(GL_DEPTH_TEST, &m_savedState.depthTest);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &m_savedState.depthMask);
    glGetBooleanv(GL_POLYGON_OFFSET_LINE, &m_savedState.polygonOffsetLine);
    glGetIntegerv(GL_POLYGON_MODE, m_savedState.polygonMode);
    glGetFloatv(GL_POINT_SIZE, &m_savedState.pointSize);
    glGetFloatv(GL_LINE_WIDTH, &m_savedState.lineWidth);
    m_savedState.lightEnabled = mLightEnabled;
}

void DrawToolGL::restoreLastState()
{
    if (m_savedState.blend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    if (m_savedState.depthTest) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    glDepthMask(m_savedState.depthMask);
    if (m_savedState.polygonOffsetLine) glEnable(GL_POLYGON_OFFSET_LINE); else glDisable(GL_POLYGON_OFFSET_LINE);
    glPolygonMode(GL_FRONT_AND_BACK, m_savedState.polygonMode[0]);
    glPointSize(m_savedState.pointSize);
    glLineWidth(m_savedState.lineWidth);
    mLightEnabled = m_savedState.lightEnabled;
}

void DrawToolGL::readPixels(int x, int y, int w, int h, float* rgb, float* z)
{
    if(rgb != nullptr)
        glReadPixels(x, y, w, h, GL_RGB, GL_FLOAT, rgb);
    if(z != nullptr)
        glReadPixels(x, y, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, z);
}

} // namespace sofa::gl
