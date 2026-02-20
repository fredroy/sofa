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
#include <sofa/gl/component/rendering3d/config.h>

#include <vector>
#include <string>
#include <sofa/gl/template.h>
#include <sofa/gl/Texture.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/visual/VisualModelImpl.h>

namespace sofa::gl::component::rendering3d
{

/**
 *  \brief Main class for rendering 3D model in SOFA.
 *
 *  This class implements VisuelModelImpl with rendering functions
 *  using OpenGL.
 *
 */
class SOFA_GL_COMPONENT_RENDERING3D_API OglModel : public sofa::component::visual::VisualModelImpl
{
public:
    using Inherit = sofa::component::visual::VisualModelImpl;

    SOFA_CLASS(OglModel, Inherit);


    Data<bool> blendTransparency; ///< Blend transparent parts
protected:
    Data<bool> premultipliedAlpha; ///< is alpha premultiplied ?
    Data<bool> writeZTransparent; ///< Write into Z Buffer for Transparent Object
    Data<bool> alphaBlend; ///< Enable alpha blending
    Data<bool> depthTest; ///< Enable depth testing
    Data<int> cullFace; ///< Face culling (0 = no culling, 1 = cull back faces, 2 = cull front faces)
    Data<GLfloat> lineWidth; ///< Line width (set if != 1, only for lines rendering)
    Data<GLfloat> pointSize; ///< Point size (set if != 1, only for points rendering)
    Data<bool> lineSmooth; ///< Enable smooth line rendering
    Data<bool> pointSmooth; ///< Enable smooth point rendering

    // primitive types
    Data<sofa::helper::OptionsGroup> primitiveType; ///< Select types of primitives to send (necessary for some shader types such as geometry or tessellation)

    //alpha blend function
    Data<sofa::helper::OptionsGroup> blendEquation; ///< if alpha blending is enabled this specifies how source and destination colors are combined
    Data<sofa::helper::OptionsGroup> sourceFactor; ///< if alpha blending is enabled this specifies how the red, green, blue, and alpha source blending factors are computed
    Data<sofa::helper::OptionsGroup> destFactor; ///< if alpha blending is enabled this specifies how the red, green, blue, and alpha destination blending factors are computed
    GLenum blendEq, sfactor, dfactor;

    sofa::gl::Texture *m_tex; //this texture is used only if a texture name is specified in the scn
    GLuint vbo, iboEdges, iboTriangles, iboQuads;
    bool VBOGenDone, initDone, useEdges, useTriangles, useQuads, canUsePatches;
    size_t oldVerticesSize, oldNormalsSize, oldTexCoordsSize, oldTangentsSize, oldBitangentsSize, oldEdgesSize, oldTrianglesSize, oldQuadsSize;
    int edgesRevision, trianglesRevision, quadsRevision;

    /// These two buffers are used to convert the data field to float type before being sent to
    /// opengl
    std::vector<sofa::type::Vec3f> verticesTmpBuffer;
    std::vector<sofa::type::Vec3f> normalsTmpBuffer;

    void internalDraw(const core::visual::VisualParams* vparams, bool transparent) override;

    void drawGroup(int ig, bool transparent);
    void drawGroups(bool transparent);

    virtual void pushTransformMatrix(float* matrix);
    virtual void popTransformMatrix();

    // Modern GL shader infrastructure
    GLuint m_oglVao = 0;
    GLuint m_oglProgram = 0;
    bool m_oglShaderReady = false;

    GLint m_uMVMatrix = -1, m_uProjMatrix = -1, m_uNormalMatrix = -1;
    GLint m_uMatAmbient = -1, m_uMatDiffuse = -1, m_uMatSpecular = -1;
    GLint m_uMatEmissive = -1, m_uMatShininess = -1;
    GLint m_uLightPos = -1, m_uLightAmb = -1, m_uLightDif = -1, m_uLightSpec = -1;
    GLint m_uHasTexture = -1, m_uTexSampler = -1;
    GLuint m_dummyTexture = 0;

    float m_baseMV[16] {};
    float m_currentMV[16] {};

    void initOglShader();
    void uploadModelViewToShader();
    static void computeNormalMatrix3x3(const float* mv, float* nm);
    static void mat4Mult(float* result, const float* a, const float* b);

    std::vector<sofa::gl::Texture*> textures;

    std::map<int, int> materialTextureIdMap; //link between a material and a texture

    GLenum getGLenum(const char* c ) const;

    OglModel();

    ~OglModel() override;
public:
    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    bool loadTexture(const std::string& filename) override;
    bool loadTextures() override;

    void initTextures();
    void doInitVisual(const core::visual::VisualParams* vparams) override;

    void init() override { VisualModelImpl::init(); }

    void updateBuffers() override;

    void deleteBuffers() override;
    void deleteTextures() override;

    bool hasTransparent() override;
    bool hasTexture();

public:
    bool isUseEdges()	{ return useEdges; }
    bool isUseTriangles()	{ return useTriangles; }
    bool isUseQuads()	{ return useQuads; }

    sofa::gl::Texture* getTex() const	{ return m_tex; }
    GLuint getVbo()	{ return vbo;	}
    GLuint getIboEdges() { return iboEdges; }
    GLuint getIboTriangles() { return iboTriangles; }
    GLuint getIboQuads()    { return iboQuads; }
    const std::vector<sofa::gl::Texture*>& getTextures() const { return textures;	}

    void createVertexBuffer();
    void createEdgesIndicesBuffer();
    void createTrianglesIndicesBuffer();
    void createQuadsIndicesBuffer();
    void initVertexBuffer();
    void initEdgesIndicesBuffer();
    void initTrianglesIndicesBuffer();
    void initQuadsIndicesBuffer();
    void updateVertexBuffer();
    void updateEdgesIndicesBuffer();
    void updateTrianglesIndicesBuffer();
    void updateQuadsIndicesBuffer();
};

} // namespace sofa::gl::component::rendering3d
