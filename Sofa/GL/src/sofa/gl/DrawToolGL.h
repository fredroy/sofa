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

#include <sofa/helper/visual/DrawTool.h>

#include <sofa/type/RGBAColor.h>
#include <sofa/gl/gl.h>
#include <sofa/gl/CoreProfileRenderer.h>

#include <vector>
#include <stack>
#include <array>

namespace sofa::gl
{

class SOFA_GL_API DrawToolGL : public helper::visual::DrawTool
{

public:
    typedef sofa::type::Quat<SReal> Quaternion;

    using DrawVertex = CoreProfileRenderer::Vertex;

    DrawToolGL();
    virtual ~DrawToolGL() override;

    void init() override;

    // Matrix setup (called by the viewer to provide camera matrices)
    void setProjectionMatrix(const double* mat16);
    void setModelViewMatrix(const double* mat16);
    void setLightPosition(float x, float y, float z, float w);
    void setLightAmbient(float r, float g, float b, float a);
    void setLightDiffuse(float r, float g, float b, float a);
    void setLightSpecular(float r, float g, float b, float a);

    void drawPoint(const type::Vec3 &p, const type::RGBAColor &c) override;
    //normal on a point is useless
    void drawPoint(const type::Vec3 &p, const type::Vec3 &n, const type::RGBAColor &c) override;
    virtual void drawPoints(const std::vector<type::Vec3> &points, float size,  const type::RGBAColor& color) override;
    virtual void drawPoints(const std::vector<type::Vec3> &points, float size, const std::vector<type::RGBAColor>& color) override;

    void drawLine(const type::Vec3 &p1, const type::Vec3 &p2, const type::RGBAColor& color) override;
    void drawInfiniteLine(const type::Vec3 &point, const type::Vec3 &direction, const type::RGBAColor& color, const bool& vanishing=false) override;
    void drawInfiniteLine(const Vec3 &point, const Vec3 &direction, const float& size, const type::RGBAColor& color, const bool& vanishing=false) override;

    virtual void drawLines(const std::vector<type::Vec3> &points, float size, const type::RGBAColor& color) override;
    virtual void drawLines(const std::vector<type::Vec3> &points, float size, const std::vector<type::RGBAColor>& colors) override;
    virtual void drawLines(const std::vector<type::Vec3> &points, const std::vector< type::Vec2i > &index, float size, const type::RGBAColor& color) override;

    virtual void drawLineStrip(const std::vector<type::Vec3> &points, float size, const type::RGBAColor& color) override;
    virtual void drawLineLoop(const std::vector<type::Vec3> &points, float size, const type::RGBAColor& color) override;

    virtual void drawDisk(float radius, double from, double to, int resolution, const type::RGBAColor& color) override;
    virtual void drawCircle(float radius, float lineThickness, int resolution, const type::RGBAColor& color) override;

    void drawTriangle(const type::Vec3 &p1,const type::Vec3 &p2,const type::Vec3 &p3,
            const type::Vec3 &normal) override;
    void drawTriangle(const type::Vec3 &p1,const type::Vec3 &p2,const type::Vec3 &p3,
            const type::Vec3 &normal, const type::RGBAColor &c) override;
    void drawTriangle(const type::Vec3 &p1,const type::Vec3 &p2,const type::Vec3 &p3,
            const type::Vec3 &normal,
            const type::RGBAColor &c1, const type::RGBAColor &c2, const type::RGBAColor &c3) override;
    void drawTriangle(const type::Vec3 &p1,const type::Vec3 &p2,const type::Vec3 &p3,
            const type::Vec3 &normal1, const type::Vec3 &normal2, const type::Vec3 &normal3,
            const type::RGBAColor &c1, const type::RGBAColor &c2, const type::RGBAColor &c3) override;
    virtual void drawTriangles(const std::vector<type::Vec3> &points, const type::RGBAColor& color) override;
    virtual void drawTriangles(const std::vector<type::Vec3> &points,
            const std::vector< type::RGBAColor > &color) override;
    virtual void drawTriangles(const std::vector<type::Vec3> &points, const type::Vec3& normal, const type::RGBAColor& color) override;
    virtual void drawTriangles(const std::vector<type::Vec3> &points,
            const std::vector< type::Vec3i > &index,
            const std::vector<type::Vec3>  &normal,
            const type::RGBAColor& color) override;
    virtual void drawTriangles(const std::vector<type::Vec3> &points,
            const std::vector< type::Vec3i > &index,
            const std::vector<type::Vec3>  &normal,
            const std::vector<type::RGBAColor>& colors) override;
    virtual void drawTriangles(const std::vector<type::Vec3> &points,
            const std::vector<type::Vec3>  &normal,
            const std::vector< type::RGBAColor > &color) override;

    virtual void drawTriangleStrip(const std::vector<type::Vec3> &points,
            const std::vector<type::Vec3>  &normal,
            const type::RGBAColor& color) override;

    virtual void drawTriangleFan(const std::vector<type::Vec3> &points,
            const std::vector<type::Vec3>  &normal,
            const type::RGBAColor& color) override;

    void drawFrame(const type::Vec3& position, const Quaternion &orientation, const type::Vec3f &size) override;
    void drawFrame(const type::Vec3& position, const Quaternion &orientation, const type::Vec3f &size, const type::RGBAColor &color) override;

    virtual void drawSpheres (const std::vector<type::Vec3> &points, const std::vector<float>& radius, const type::RGBAColor& color) override;
    virtual void drawSpheres (const std::vector<type::Vec3> &points, float radius, const type::RGBAColor& color) override;
    virtual void drawFakeSpheres(const std::vector<type::Vec3> &points, const std::vector<float>& radius, const type::RGBAColor& color) override;
    virtual void drawFakeSpheres(const std::vector<type::Vec3> &points, float radius, const type::RGBAColor& color) override;

    void drawCone    (const type::Vec3& p1, const type::Vec3 &p2, float radius1, float radius2, const type::RGBAColor& color, int subd=16) override;

    void drawCube    (const float& radius, const type::RGBAColor& color, const int& subd=16) override;

    void drawCylinder(const type::Vec3& p1, const type::Vec3 &p2, float radius, const type::RGBAColor& color,  int subd=16) override;

    void drawCapsule(const type::Vec3& p1, const type::Vec3 &p2, float radius, const type::RGBAColor& color,  int subd=16) override;

    void drawArrow   (const type::Vec3& p1, const type::Vec3 &p2, float radius, const type::RGBAColor& color,  int subd=16) override;
    void drawArrow   (const type::Vec3& p1, const type::Vec3 &p2, float radius, float coneLength, const type::RGBAColor& color,  int subd=16) override;
    void drawArrow   (const type::Vec3& p1, const type::Vec3 &p2, float radius, float coneLength, float coneRadius, const type::RGBAColor& color,  int subd=16) override;

    void drawCross(const type::Vec3&p, float length, const type::RGBAColor& color) override;

    void drawPlus    (const float& radius, const type::RGBAColor& color, const int& subd=16) override;

    void drawQuad(const type::Vec3 &p1,const type::Vec3 &p2,const type::Vec3 &p3,const type::Vec3 &p4,
            const type::Vec3 &normal) override;
    void drawQuad(const type::Vec3 &p1,const type::Vec3 &p2,const type::Vec3 &p3,const type::Vec3 &p4,
            const type::Vec3 &normal, const type::RGBAColor &c) override;
    void drawQuad(const type::Vec3 &p1,const type::Vec3 &p2,const type::Vec3 &p3,const type::Vec3 &p4,
            const type::Vec3 &normal,
            const type::RGBAColor &c1, const type::RGBAColor &c2, const type::RGBAColor &c3, const type::RGBAColor &c4) override;
    void drawQuad(const type::Vec3 &p1,const type::Vec3 &p2,const type::Vec3 &p3,const type::Vec3 &p4,
            const type::Vec3 &normal1, const type::Vec3 &normal2, const type::Vec3 &normal3, const type::Vec3 &normal4,
            const type::RGBAColor &c1, const type::RGBAColor &c2, const type::RGBAColor &c3, const type::RGBAColor &c4) override;
    virtual void drawQuads(const std::vector<type::Vec3> &points, const type::RGBAColor& color) override;
    virtual void drawQuads(const std::vector<type::Vec3> &points, const std::vector<type::RGBAColor>& colors) override;


    void drawTetrahedron(const type::Vec3 &p0, const type::Vec3 &p1, const type::Vec3 &p2, const type::Vec3 &p3, const type::RGBAColor &color) override;
    void drawScaledTetrahedron(const type::Vec3& p0, const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& p3, const type::RGBAColor& color, const float scale) override;
    virtual void drawTetrahedra(const std::vector<type::Vec3> &points, const type::RGBAColor& color) override;
    virtual void drawScaledTetrahedra(const std::vector<type::Vec3> &points, const type::RGBAColor& color, const float scale) override;

    void drawHexahedron(const type::Vec3 &p0, const type::Vec3 &p1, const type::Vec3 &p2, const type::Vec3 &p3,
        const type::Vec3 &p4, const type::Vec3 &p5, const type::Vec3 &p6, const type::Vec3 &p7, const type::RGBAColor &color) override;
    virtual void drawHexahedra(const std::vector<type::Vec3> &points, const type::RGBAColor& color) override;
    virtual void drawScaledHexahedra(const std::vector<type::Vec3> &points, const type::RGBAColor& color, const float scale) override;

    void drawSphere( const type::Vec3 &p, float radius) override;
    void drawSphere(const type::Vec3 &p, float radius, const type::RGBAColor &color) override;
    void drawEllipsoid(const type::Vec3 &p, const type::Vec3 &radii) override;

    void drawBoundingBox( const type::Vec3 &min, const type::Vec3 &max, float size = 1.0) override;

    void draw3DText(const type::Vec3 &p, float scale, const type::RGBAColor &color, const char* text) override;

    virtual void draw3DText_Indices(const std::vector<type::Vec3> &positions, float scale, const type::RGBAColor &color) override;

    void clear() override;

    void setMaterial(const type::RGBAColor &color) override;

    void resetMaterial(const type::RGBAColor &color) override;
    void resetMaterial() override;

    void pushMatrix() override;
    void popMatrix() override;
    void multMatrix(float* glTransform ) override;
    void scale( float s ) override;
    void translate(float x, float y, float z) override;

    void writeOverlayText( int x, int y, unsigned fontSize, const type::RGBAColor &color, const char* text ) override;

    void enablePolygonOffset(float factor, float units) override;
    void disablePolygonOffset() override;

    void enableBlending() override;
    void disableBlending() override;

    void enableLighting() override;
    void disableLighting() override;

    void enableDepthTest() override;
    void disableDepthTest() override;

    void saveLastState() override;
    void restoreLastState() override;

    void readPixels(int x, int y, int w, int h, float* rgb, float* z = nullptr) override;

protected:

    bool mLightEnabled;
    int  mPolygonMode;      //0: no cull, 1 front (CULL_CLOCKWISE), 2 back (CULL_ANTICLOCKWISE)
    bool mWireFrameEnabled;

    // Vertex buffer (fed to CoreProfileRenderer)
    std::vector<DrawVertex> m_vertexBuffer;

    // Internal matrix stack (replaces legacy GL matrix stack)
    using Mat16f = std::array<float, 16>;
    Mat16f m_projectionMatrix;
    std::stack<Mat16f> m_modelViewStack;

    // Internal light state (replaces glGetLightfv)
    float m_lightPos[4]  = {0.0f, 0.0f, 1.0f, 0.0f};
    float m_lightAmb[4]  = {0.2f, 0.2f, 0.2f, 1.0f};
    float m_lightDif[4]  = {0.8f, 0.8f, 0.8f, 1.0f};
    float m_lightSpec[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // Saved state for saveLastState/restoreLastState
    struct SavedGLState {
        GLboolean blend;
        GLboolean depthTest;
        GLboolean depthMask;
        GLboolean polygonOffsetLine;
        GLint polygonMode[2];
        GLfloat pointSize;
        GLfloat lineWidth;
        bool lightEnabled;
    };
    SavedGLState m_savedState {};

    void flushVertexBuffer(GLenum mode, bool enableLighting = true);

    void pushVertex(const type::Vec3& pos, const type::Vec3& normal, const type::RGBAColor& color);
    void pushVertex(float px, float py, float pz, float nx, float ny, float nz, const type::RGBAColor& color);
    void pushQuadAsTriangles(const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& p3, const type::Vec3& p4,
            const type::Vec3& normal, const type::RGBAColor& color);
    void pushQuadAsTriangles(const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& p3, const type::Vec3& p4,
            const type::Vec3& normal,
            const type::RGBAColor& c1, const type::RGBAColor& c2, const type::RGBAColor& c3, const type::RGBAColor& c4);
    void pushQuadAsTriangles(const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& p3, const type::Vec3& p4,
            const type::Vec3& n1, const type::Vec3& n2, const type::Vec3& n3, const type::Vec3& n4,
            const type::RGBAColor& c1, const type::RGBAColor& c2, const type::RGBAColor& c3, const type::RGBAColor& c4);

    // Matrix math helpers
    static void mat4Identity(float* m);
    static void mat4Multiply(float* result, const float* a, const float* b);
    static void mat4Translation(float* m, float x, float y, float z);
    static void mat4Scale(float* m, float sx, float sy, float sz);
    const Mat16f& currentModelView() const;

    // Sphere generation
    void generateSphereTriangles(const type::Vec3& center, float rx, float ry, float rz,
                                  const type::RGBAColor& color, unsigned int rings, unsigned int sectors);

public:
    // getter & setter
    void setLightingEnabled(bool _isAnabled) override;

    bool getLightEnabled() {return mLightEnabled;}

    const float* getLightPosition() const { return m_lightPos; }
    const float* getLightAmbient() const { return m_lightAmb; }
    const float* getLightDiffuse() const { return m_lightDif; }
    const float* getLightSpecular() const { return m_lightSpec; }

    void setPolygonMode(int _mode, bool _wireframe) override;

    int getPolygonMode() {return mPolygonMode;}
    bool getWireFrameEnabled() {return mWireFrameEnabled;}
};

}//namespace sofa::gl
