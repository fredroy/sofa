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
#include <sofa/gl/component/rendering3d/OglModel.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/gl/gl.h>
#include <sofa/gl/RAII.h>
#include <sofa/type/vector.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <cstring>
#include <sofa/type/RGBAColor.h>
#include <sofa/gl/DrawToolGL.h>

namespace sofa::gl::component::rendering3d
{

using sofa::type::RGBAColor;
using sofa::type::Material;
using namespace sofa::type;

// ---------------------------------------------------------------------------
// GLSL 410 core shaders for OglModel Blinn-Phong rendering
// ---------------------------------------------------------------------------
static const char* s_oglVertexShader = R"GLSL(
#version 410 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texCoord;

uniform mat4 u_modelViewMatrix;
uniform mat4 u_projectionMatrix;
uniform mat3 u_normalMatrix;

out vec3 v_viewPos;
out vec3 v_normal;
out vec2 v_texCoord;

void main()
{
    vec4 viewPos4 = u_modelViewMatrix * vec4(a_position, 1.0);
    v_viewPos  = viewPos4.xyz;
    v_normal   = u_normalMatrix * a_normal;
    v_texCoord = a_texCoord;
    gl_Position = u_projectionMatrix * viewPos4;
    gl_PointSize = 1.0;
}
)GLSL";

static const char* s_oglFragmentShader = R"GLSL(
#version 410 core

in vec3 v_viewPos;
in vec3 v_normal;
in vec2 v_texCoord;

uniform vec4  u_matAmbient;
uniform vec4  u_matDiffuse;
uniform vec4  u_matSpecular;
uniform vec4  u_matEmissive;
uniform float u_matShininess;

uniform vec4 u_lightPosition;
uniform vec4 u_lightAmbient;
uniform vec4 u_lightDiffuse;
uniform vec4 u_lightSpecular;

uniform bool      u_hasTexture;
uniform sampler2D u_texture;

out vec4 fragColor;

void main()
{
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing)
        N = -N;

    // Light direction (view space). w==0 means directional.
    vec3 L;
    if (u_lightPosition.w == 0.0)
        L = normalize(u_lightPosition.xyz);
    else
        L = normalize(u_lightPosition.xyz - v_viewPos);

    float NdotL = max(dot(N, L), 0.0);

    // View direction (camera is at origin in view space)
    vec3 V = normalize(-v_viewPos);
    // Blinn half-vector
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float spec = (NdotL > 0.0) ? pow(NdotH, u_matShininess) : 0.0;

    vec4 ambient  = u_lightAmbient  * u_matAmbient;
    vec4 diffuse  = u_lightDiffuse  * u_matDiffuse  * NdotL;
    vec4 specular = u_lightSpecular * u_matSpecular * spec;
    vec4 color    = u_matEmissive + ambient + diffuse + specular;

    if (u_hasTexture)
    {
        vec4 texColor = texture(u_texture, v_texCoord);
        color *= texColor;
    }

    color.a = u_matDiffuse.a;
    fragColor = clamp(color, 0.0, 1.0);
}
)GLSL";

void registerOglModel(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Generic visual model for OpenGL display.")
        .add< OglModel >());
}

OglModel::OglModel()
    : blendTransparency(initData(&blendTransparency, true, "blendTranslucency", "Blend transparent parts"))
    , premultipliedAlpha(initData(&premultipliedAlpha, false, "premultipliedAlpha", "is alpha premultiplied ?"))
    , writeZTransparent(initData(&writeZTransparent, false, "writeZTransparent", "Write into Z Buffer for Transparent Object"))
    , alphaBlend(initData(&alphaBlend, false, "alphaBlend", "Enable alpha blending"))
    , depthTest(initData(&depthTest, true, "depthTest", "Enable depth testing"))
    , cullFace(initData(&cullFace, 0, "cullFace", "Face culling (0 = no culling, 1 = cull back faces, 2 = cull front faces)"))
    , lineWidth(initData(&lineWidth, 1.0f, "lineWidth", "Line width (set if != 1, only for lines rendering)"))
    , pointSize(initData(&pointSize, 1.0f, "pointSize", "Point size (set if != 1, only for points rendering)"))
    , lineSmooth(initData(&lineSmooth, false, "lineSmooth", "Enable smooth line rendering"))
    , pointSmooth(initData(&pointSmooth, false, "pointSmooth", "Enable smooth point rendering"))
    , primitiveType( initData(&primitiveType, "primitiveType", "Select types of primitives to send (necessary for some shader types such as geometry or tessellation)"))
    , blendEquation( initData(&blendEquation, "blendEquation", "if alpha blending is enabled this specifies how source and destination colors are combined") )
    , sourceFactor( initData(&sourceFactor, "sfactor", "if alpha blending is enabled this specifies how the red, green, blue, and alpha source blending factors are computed") )
    , destFactor( initData(&destFactor, "dfactor", "if alpha blending is enabled this specifies how the red, green, blue, and alpha destination blending factors are computed") )
    , m_tex(nullptr)
    , vbo(0), iboEdges(0), iboTriangles(0), iboQuads(0)
    , VBOGenDone(false), initDone(false), useEdges(false), useTriangles(false), useQuads(false), canUsePatches(false)
    , oldVerticesSize(0), oldNormalsSize(0), oldTexCoordsSize(0), oldTangentsSize(0), oldBitangentsSize(0), oldEdgesSize(0), oldTrianglesSize(0), oldQuadsSize(0)
    , edgesRevision(-1), trianglesRevision(-1), quadsRevision(-1)
{

    textures.clear();

    blendEquation.setValue({"GL_FUNC_ADD", "GL_FUNC_SUBTRACT", "GL_MIN", "GL_MAX"});
    sourceFactor.setValue(helper::OptionsGroup{"GL_ZERO", "GL_ONE", "GL_SRC_ALPHA", "GL_ONE_MINUS_SRC_ALPHA"}.setSelectedItem(2));
    destFactor.setValue(helper::OptionsGroup{"GL_ZERO", "GL_ONE", "GL_SRC_ALPHA", "GL_ONE_MINUS_SRC_ALPHA"}.setSelectedItem(3));
    primitiveType.setValue(helper::OptionsGroup{"DEFAULT", "LINES_ADJACENCY", "PATCHES", "POINTS"}.setSelectedItem(0));
}

void OglModel::parse(core::objectmodel::BaseObjectDescription* arg)
{
    if (arg->getAttribute("isEnabled"))
    {
        msg_warning() << "isEnabled field has been renamed to \'enabled\' since v23.12 (#3931).";

        this->d_enable.setValue(std::strcmp(arg->getAttribute("isEnabled"), "true") == 0 || arg->getAttributeAsInt("isEnabled"));
    }

    Inherit::parse(arg);
}

void OglModel::deleteTextures()
{
    if (m_tex != nullptr) 
    {
        delete m_tex;
        m_tex = nullptr;
    }

    for (unsigned int i = 0 ; i < textures.size() ; i++)
    {
        delete textures[i];
        textures[i] = nullptr;
    }
}

void OglModel::deleteBuffers()
{
    // NB fjourdes : I don t know why gDEBugger still reports
    // graphics memory leaks after destroying the GLContext
    // even if the vbos destruction is claimed with the following
    // lines...
    if( vbo > 0 )
    {
        glDeleteBuffers(1,&vbo);
    }
    if( iboEdges > 0)
    {
        glDeleteBuffers(1,&iboEdges);
    }
    if( iboTriangles > 0)
    {
        glDeleteBuffers(1,&iboTriangles);
    }
    if( iboQuads > 0 )
    {
        glDeleteBuffers(1,&iboQuads);
    }
}

OglModel::~OglModel()
{
    if (m_oglVao) glDeleteVertexArrays(1, &m_oglVao);
    if (m_oglProgram) glDeleteProgram(m_oglProgram);
    if (m_dummyTexture) glDeleteTextures(1, &m_dummyTexture);
    deleteTextures();
    deleteBuffers();
}

// ---------------------------------------------------------------------------
// Modern GL shader infrastructure
// ---------------------------------------------------------------------------

void OglModel::initOglShader()
{
    if (m_oglShaderReady)
        return;

    auto compileShader = [this](GLenum type, const char* src) -> GLuint
    {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            char log[1024];
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            msg_error() << "OglModel shader compile error ("
                        << (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
                        << "):\n" << log;
            glDeleteShader(s);
            return 0;
        }
        return s;
    };

    GLuint vs = compileShader(GL_VERTEX_SHADER, s_oglVertexShader);
    if (!vs) return;

    GLuint fs = compileShader(GL_FRAGMENT_SHADER, s_oglFragmentShader);
    if (!fs) { glDeleteShader(vs); return; }

    m_oglProgram = glCreateProgram();
    glAttachShader(m_oglProgram, vs);
    glAttachShader(m_oglProgram, fs);
    glLinkProgram(m_oglProgram);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = 0;
    glGetProgramiv(m_oglProgram, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        char log[1024];
        glGetProgramInfoLog(m_oglProgram, sizeof(log), nullptr, log);
        msg_error() << "OglModel shader link error:\n" << log;
        glDeleteProgram(m_oglProgram);
        m_oglProgram = 0;
        return;
    }

    // Cache uniform locations
    m_uMVMatrix    = glGetUniformLocation(m_oglProgram, "u_modelViewMatrix");
    m_uProjMatrix  = glGetUniformLocation(m_oglProgram, "u_projectionMatrix");
    m_uNormalMatrix= glGetUniformLocation(m_oglProgram, "u_normalMatrix");
    m_uMatAmbient  = glGetUniformLocation(m_oglProgram, "u_matAmbient");
    m_uMatDiffuse  = glGetUniformLocation(m_oglProgram, "u_matDiffuse");
    m_uMatSpecular = glGetUniformLocation(m_oglProgram, "u_matSpecular");
    m_uMatEmissive = glGetUniformLocation(m_oglProgram, "u_matEmissive");
    m_uMatShininess= glGetUniformLocation(m_oglProgram, "u_matShininess");
    m_uLightPos    = glGetUniformLocation(m_oglProgram, "u_lightPosition");
    m_uLightAmb    = glGetUniformLocation(m_oglProgram, "u_lightAmbient");
    m_uLightDif    = glGetUniformLocation(m_oglProgram, "u_lightDiffuse");
    m_uLightSpec   = glGetUniformLocation(m_oglProgram, "u_lightSpecular");
    m_uHasTexture  = glGetUniformLocation(m_oglProgram, "u_hasTexture");
    m_uTexSampler  = glGetUniformLocation(m_oglProgram, "u_texture");

    // Create VAO
    glGenVertexArrays(1, &m_oglVao);

    // Create a 1x1 white dummy texture so the sampler always has a valid texture bound
    glGenTextures(1, &m_dummyTexture);
    glBindTexture(GL_TEXTURE_2D, m_dummyTexture);
    const GLubyte white[] = { 255, 255, 255, 255 };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    m_oglShaderReady = true;
}

void OglModel::computeNormalMatrix3x3(const float* mv, float* nm)
{
    // Transpose of inverse of upper-left 3x3 of the modelview matrix
    const float a00 = mv[0], a01 = mv[4], a02 = mv[8];
    const float a10 = mv[1], a11 = mv[5], a12 = mv[9];
    const float a20 = mv[2], a21 = mv[6], a22 = mv[10];

    const float det = a00 * (a11 * a22 - a12 * a21)
                    - a01 * (a10 * a22 - a12 * a20)
                    + a02 * (a10 * a21 - a11 * a20);

    if (std::fabs(det) < 1e-12f)
    {
        // Degenerate matrix - use identity
        nm[0] = 1; nm[1] = 0; nm[2] = 0;
        nm[3] = 0; nm[4] = 1; nm[5] = 0;
        nm[6] = 0; nm[7] = 0; nm[8] = 1;
        return;
    }

    const float invDet = 1.0f / det;
    // Cofactor matrix transposed (= adjugate), then divided by det
    // Result is transpose(inverse(M3x3))
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

void OglModel::mat4Mult(float* result, const float* a, const float* b)
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

void OglModel::uploadModelViewToShader()
{
    glUniformMatrix4fv(m_uMVMatrix, 1, GL_FALSE, m_currentMV);

    float normalMatrix[9];
    computeNormalMatrix3x3(m_currentMV, normalMatrix);
    glUniformMatrix3fv(m_uNormalMatrix, 1, GL_TRUE, normalMatrix);
}

void OglModel::pushTransformMatrix(float* matrix)
{
    mat4Mult(m_currentMV, m_baseMV, matrix);
    uploadModelViewToShader();
}

void OglModel::popTransformMatrix()
{
    std::copy(m_baseMV, m_baseMV + 16, m_currentMV);
    uploadModelViewToShader();
}

// ---------------------------------------------------------------------------

void OglModel::drawGroup(int ig, bool transparent)
{
    const Inherit::VecVisualEdge& edges = this->getEdges();
    const Inherit::VecVisualTriangle& triangles = this->getTriangles();
    const Inherit::VecVisualQuad& quads = this->getQuads();

    const VecCoord& vertices = this->getVertices();

    FaceGroup g;
    if (ig < 0)
    {
        g.materialId = -1;
        g.edge0 = 0;
        g.nbe = int(edges.size());
        g.tri0 = 0;
        g.nbt = int(triangles.size());
        g.quad0 = 0;
        g.nbq = int(quads.size());
    }
    else
    {
        g = this->d_groups.getValue()[size_t(ig)];
    }
    Material m;
    if (g.materialId < 0)
        m = this->d_material.getValue();
    else
        m = this->d_materials.getValue()[size_t(g.materialId)];

    bool isTransparent = (m.useDiffuse && m.diffuse[3] < 1.0f) || hasTransparent();
    if (transparent ^ isTransparent) return;

    // Per-group texture binding (texture coordinates are already set up in VAO)
    if (!m_tex && m.useTexture && m.activated)
    {
        //get the texture id corresponding to the current material
        size_t indexInTextureArray = size_t(materialTextureIdMap[g.materialId]);
        if (indexInTextureArray < textures.size() && textures[indexInTextureArray])
        {
            glActiveTexture(GL_TEXTURE0);
            textures[indexInTextureArray]->bind();
        }
        glUniform1i(m_uHasTexture, 1);
    }

    RGBAColor ambient = m.useAmbient?m.ambient:RGBAColor::black();
    RGBAColor diffuse = m.useDiffuse?m.diffuse:RGBAColor::black();
    RGBAColor specular = m.useSpecular?m.specular:RGBAColor::black();
    RGBAColor emissive = m.useEmissive?m.emissive:RGBAColor::black();
    float shininess = m.useShininess?m.shininess:45;
    if( shininess > 128.0f ) shininess = 128.0f;

    if (shininess == 0.0f)
    {
        specular = RGBAColor::black() ;
        shininess = 1;
    }

    if (isTransparent)
    {
        emissive[3] = 0;
        ambient[3] = 0;
        specular[3] = 0;
    }

    const bool drawPoints = (primitiveType.getValue().getSelectedId() == 3);
    if (drawPoints)
    {
        // Simulate unlit rendering: put diffuse color into emissive, zero the rest
        glUniform4fv(m_uMatAmbient, 1, RGBAColor::black().data());
        glUniform4fv(m_uMatDiffuse, 1, RGBAColor::black().data());
        glUniform4fv(m_uMatSpecular, 1, RGBAColor::black().data());
        glUniform4fv(m_uMatEmissive, 1, diffuse.data());
        glUniform1f(m_uMatShininess, 1.0f);

        glDrawArrays(GL_POINTS, 0, GLsizei(vertices.size()));

        // Restore actual material for subsequent geometry if any
        glUniform4fv(m_uMatEmissive, 1, emissive.data());
    }
    else
    {
        // Upload material uniforms
        glUniform4fv(m_uMatAmbient, 1, ambient.data());
        glUniform4fv(m_uMatDiffuse, 1, diffuse.data());
        glUniform4fv(m_uMatSpecular, 1, specular.data());
        glUniform4fv(m_uMatEmissive, 1, emissive.data());
        glUniform1f(m_uMatShininess, shininess);
    }

    if (g.nbe > 0 && !drawPoints)
    {
        const VisualEdge* indices = nullptr;

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboEdges);

        GLenum prim = GL_LINES;
        switch (primitiveType.getValue().getSelectedId())
        {
        case 1:
            msg_warning() << "LINES_ADJACENCY primitive type invalid for edge topologies" ;
            break;
        case 2:
            prim = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES,2);
            break;
        default:
            break;
        }

        glDrawElements(prim, g.nbe * 2, GL_UNSIGNED_INT, indices + g.edge0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    if (g.nbt > 0 && !drawPoints)
    {
        const VisualTriangle* indices = nullptr;

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);

        GLenum prim = GL_TRIANGLES;
        switch (primitiveType.getValue().getSelectedId())
        {
        case 1:
            msg_warning() << "LINES_ADJACENCY primitive type invalid for triangular topologies" ;
            break;
        case 2:
            prim = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES,3);
            break;
        default:
            break;
        }

        glDrawElements(prim, g.nbt * 3, GL_UNSIGNED_INT, indices + g.tri0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    if (g.nbq > 0 && !drawPoints)
    {
        const VisualQuad* indices = nullptr;

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);

        GLenum prim = GL_QUADS;
        switch (primitiveType.getValue().getSelectedId())
        {
        case 1:
            prim = GL_LINES_ADJACENCY_EXT;
            break;
        case 2:
            prim = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES,4);
            break;
        default:
            break;
        }

        glDrawElements(prim, g.nbq * 4, GL_UNSIGNED_INT, indices + g.quad0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    if (!m_tex && m.useTexture && m.activated)
    {
        int indexInTextureArray = materialTextureIdMap[g.materialId];
        if (indexInTextureArray < int(textures.size()) && textures[size_t(indexInTextureArray)])
        {
            textures[size_t(indexInTextureArray)]->unbind();
        }
        glBindTexture(GL_TEXTURE_2D, m_dummyTexture);
        glUniform1i(m_uHasTexture, 0);
    }
}

void OglModel::drawGroups(bool transparent)
{
    const helper::ReadAccessor< Data< type::vector<FaceGroup> > > groups = this->d_groups;

    if (groups.empty())
    {
        drawGroup(-1, transparent);
    }
    else
    {
        for (size_t i=0; i<groups.size(); ++i)
            drawGroup(int(i), transparent);
    }
}


template<class InType, class OutType>
void copyVector(const InType& src, OutType& dst)
{
    unsigned int i=0;
    for(auto& item : src)
    {
        dst[i].set(item);
        ++i;
    }
}

void OglModel::internalDraw(const core::visual::VisualParams* vparams, bool transparent)
{
    if (!vparams->displayFlags().getShowVisualModels())
        return;

    /// Checks that the VBO's are ready.
    if(!VBOGenDone)
        return;

    // Lazy-init the modern GL shader and VAO
    initOglShader();
    if (!m_oglShaderReady)
        return;

    const VecCoord& vertices = this->getVertices();
    const VecDeriv& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();

    /// Force the data to be of float type before sending to opengl...
    const GLuint vertexdatasize = sizeof(Vec3f);
    const GLuint normaldatasize = sizeof(Vec3f);

    const GLulong vertexArrayByteSize = vertices.size() * vertexdatasize;
    const GLulong normalArrayByteSize = vnormals.size() * normaldatasize;

    // --- Read matrices from vparams (double -> float) ---
    {
        double dMV[16];
        vparams->getModelViewMatrix(dMV);
        for (int i = 0; i < 16; ++i)
        {
            m_baseMV[i]    = static_cast<float>(dMV[i]);
            m_currentMV[i] = m_baseMV[i];
        }
    }

    float projMatrix[16];
    {
        double dProj[16];
        vparams->getProjectionMatrix(dProj);
        for (int i = 0; i < 16; ++i)
            projMatrix[i] = static_cast<float>(dProj[i]);
    }

    // --- Read light state from DrawToolGL if available, else use defaults ---
    float lightPos[4]  = {0.0f, 0.5f, 1.0f, 0.0f};
    float lightAmb[4]  = {0.2f, 0.2f, 0.2f, 1.0f};
    float lightDif[4]  = {0.8f, 0.8f, 0.8f, 1.0f};
    float lightSpec[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    if (const auto* drawToolGL = dynamic_cast<const sofa::gl::DrawToolGL*>(vparams->drawTool()))
    {
        std::copy(drawToolGL->getLightPosition(), drawToolGL->getLightPosition() + 4, lightPos);
        std::copy(drawToolGL->getLightAmbient(),  drawToolGL->getLightAmbient()  + 4, lightAmb);
        std::copy(drawToolGL->getLightDiffuse(),  drawToolGL->getLightDiffuse()  + 4, lightDif);
        std::copy(drawToolGL->getLightSpecular(), drawToolGL->getLightSpecular() + 4, lightSpec);
    }

    // --- Bind VAO and set up vertex attribute pointers ---
    glBindVertexArray(m_oglVao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Attrib 0: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    // Attrib 1: normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(vertexArrayByteSize));
    glEnableVertexAttribArray(1);

    // Attrib 2: texcoord (optional)
    const bool hasTexCoords = (m_tex || d_putOnlyTexCoords.getValue() || !textures.empty());
    if (hasTexCoords && !vtexcoords.empty())
    {
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0,
                              reinterpret_cast<void*>(vertexArrayByteSize + normalArrayByteSize));
        glEnableVertexAttribArray(2);
    }

    // Note: do NOT unbind VBO here - VAO records the bound VBO for each attribute

    // --- Activate shader ---
    glUseProgram(m_oglProgram);

    // Upload projection matrix
    glUniformMatrix4fv(m_uProjMatrix, 1, GL_FALSE, projMatrix);

    // Upload modelview + normal matrix
    uploadModelViewToShader();

    // Upload light uniforms
    glUniform4fv(m_uLightPos,  1, lightPos);
    glUniform4fv(m_uLightAmb,  1, lightAmb);
    glUniform4fv(m_uLightDif,  1, lightDif);
    glUniform4fv(m_uLightSpec, 1, lightSpec);

    // Texture sampler is always unit 0
    glUniform1i(m_uTexSampler, 0);

    // Always bind a valid texture to unit 0 to avoid driver warnings
    glActiveTexture(GL_TEXTURE0);
    if (m_tex)
    {
        m_tex->bind();
        glUniform1i(m_uHasTexture, 1);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, m_dummyTexture);
        glUniform1i(m_uHasTexture, 0);
    }

    // --- Blending for transparency (first pass: subtractive) ---
    if (transparent && blendTransparency.getValue())
    {
        glEnable(GL_BLEND);
        if (writeZTransparent.getValue())
            glDepthMask(GL_TRUE);
        else glDepthMask(GL_FALSE);

        glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);

        drawGroups(transparent);

        if (premultipliedAlpha.getValue())
            glBlendFunc(GL_ONE, GL_ONE);
        else
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    }

    if (alphaBlend.getValue())
    {
        glDepthMask(GL_FALSE);
        glBlendEquation( blendEq );
        glBlendFunc( sfactor, dfactor );
        glEnable(GL_BLEND);
    }

    if (!depthTest.getValue())
        glDisable(GL_DEPTH_TEST);

    switch (cullFace.getValue())
    {
    case 1:
        glCullFace(GL_BACK);
        glEnable(GL_CULL_FACE);
        break;
    case 2:
        glCullFace(GL_FRONT);
        glEnable(GL_CULL_FACE);
        break;
    }

    if (lineWidth.isSet())
    {
        glLineWidth(lineWidth.getValue());
    }

    if (pointSize.isSet())
    {
        glPointSize(pointSize.getValue());
    }

    drawGroups(transparent);

    if (lineWidth.isSet())
    {
        glLineWidth(1.0f);
    }

    if (pointSize.isSet())
    {
        glPointSize(1.0f);
    }

    switch (cullFace.getValue())
    {
    case 1:
    case 2:
        glDisable(GL_CULL_FACE);
        break;
    }

    if (!depthTest.getValue())
        glEnable(GL_DEPTH_TEST);

    if (alphaBlend.getValue())
    {
        // restore Default value
        glBlendEquation( GL_FUNC_ADD );
        glBlendFunc( GL_ONE, GL_ONE );
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }

    // Unbind global texture if set
    if (m_tex)
    {
        m_tex->unbind();
    }

    if (transparent && blendTransparency.getValue())
    {
        glDisable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_TRUE);
    }

    // --- Deactivate shader and unbind VAO ---
    glUseProgram(0);

    // Disable vertex attribs
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    if (hasTexCoords && !vtexcoords.empty())
        glDisableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Normal visualization is currently a no-op in core profile.
    // TODO: Implement normal visualization with a separate line-drawing shader.
    // if (vparams->displayFlags().getShowNormals()) { ... }
}

bool OglModel::hasTransparent()
{
    if(alphaBlend.getValue())
        return true;
    return VisualModelImpl::hasTransparent();
}

bool OglModel::hasTexture()
{
    return !textures.empty() || m_tex;
}

bool OglModel::loadTexture(const std::string& filename)
{
    helper::io::Image *img = helper::io::Image::Create(filename);
    if (!img)
        return false;
    m_tex = new sofa::gl::Texture(img, true, true, false, d_srgbTexturing.getValue());
    return true;
}

// a false result indicated problems during textures loading
bool OglModel::loadTextures()
{
    bool result = true;
    textures.clear();

    //count the total number of activated textures
    std::vector<unsigned int> activatedTextures;
    for (unsigned int i = 0 ; i < this->d_materials.getValue().size() ; ++i)
        if (this->d_materials.getValue()[i].useTexture && this->d_materials.getValue()[i].activated)
            activatedTextures.push_back(i);

    for (std::vector< unsigned int>::iterator i = activatedTextures.begin() ; i < activatedTextures.end(); ++i)
    {
        std::string textureFile(this->d_materials.getValue()[*i].textureFilename);

        if (!sofa::helper::system::DataRepository.findFile(textureFile))
        {
            textureFile = this->d_fileMesh.getFullPath();
            const std::size_t position = textureFile.rfind("/");
            textureFile.replace (position+1,textureFile.length() - position, this->d_materials.getValue()[*i].textureFilename);

            if (!sofa::helper::system::DataRepository.findFile(textureFile))
            {
                msg_error() << "Texture \"" << this->d_materials.getValue()[*i].textureFilename << "\" not found"
                            << " in material " << this->d_materials.getValue()[*i].name ;
                result = false;
                continue;
            }
        }

        helper::io::Image *img = helper::io::Image::Create(textureFile);
        if (!img)
        {
            msg_error() << "couldn't create an image from file " << this->d_materials.getValue()[*i].textureFilename ;
            result = false;
            continue;
        }
        sofa::gl::Texture * text = new sofa::gl::Texture(img, true, true, false, d_srgbTexturing.getValue());
        materialTextureIdMap.insert(std::pair<int, int>(*i,textures.size()));
        textures.push_back( text );
    }

    if (textures.size() != activatedTextures.size())
        msg_error() << (activatedTextures.size() - textures.size()) << " textures couldn't be loaded" ;

    return result;
}

void OglModel::doInitVisual(const core::visual::VisualParams*)
{
    initTextures();

    initDone = true;

    if (primitiveType.getValue().getSelectedId() == 1 && !GLEW_EXT_geometry_shader4)
    {
        msg_warning() << "GL_EXT_geometry_shader4 not supported by your graphics card and/or OpenGL driver." ;
    }

    canUsePatches = (glewIsSupported("GL_ARB_tessellation_shader")!=0);

    if (primitiveType.getValue().getSelectedId() == 2 && !canUsePatches)
    {
        msg_warning() << "GL_ARB_tessellation_shader not supported by your graphics card and/or OpenGL driver." ;
        msg_warning() << "GL Version: " << glGetString(GL_VERSION) ;
        msg_warning() << "GL Vendor : " << glGetString(GL_VENDOR) ;
        msg_warning() << "GL Extensions: " << glGetString(GL_EXTENSIONS) ;
    }

    updateBuffers();

    // forcing the normal computation if we do not want to use the given ones
    if( !this->d_useNormals.getValue() ) { this->m_vnormals.beginWriteOnly()->clear(); this->m_vnormals.endEdit(); }
    computeNormals();

    if (d_updateTangents.getValue())
        computeTangents();

    if ( alphaBlend.getValue() )
    {
        blendEq = getGLenum( blendEquation.getValue().getSelectedItem().c_str() );
        sfactor = getGLenum( sourceFactor.getValue().getSelectedItem().c_str() );
        dfactor = getGLenum( destFactor.getValue().getSelectedItem().c_str() );
    }
}

void OglModel::initTextures()
{
    if (m_tex)
    {
        m_tex->init();
    }
    else
    {
        if (!textures.empty())
        {
            for (unsigned int i = 0 ; i < textures.size() ; i++)
            {
                textures[i]->init();
            }
        }
    }
}

void OglModel::createVertexBuffer()
{
    glGenBuffers(1, &vbo);
    initVertexBuffer();
    VBOGenDone = true;
}

void OglModel::createEdgesIndicesBuffer()
{
    glGenBuffers(1, &iboEdges);
    initEdgesIndicesBuffer();
    useEdges = true;
}

void OglModel::createTrianglesIndicesBuffer()
{
    glGenBuffers(1, &iboTriangles);
    initTrianglesIndicesBuffer();
    useTriangles = true;
}


void OglModel::createQuadsIndicesBuffer()
{
    glGenBuffers(1, &iboQuads);
    initQuadsIndicesBuffer();
    useQuads = true;
}

void OglModel::initVertexBuffer()
{
    size_t positionsBufferSize, normalsBufferSize;
    size_t textureCoordsBufferSize = 0, tangentsBufferSize = 0, bitangentsBufferSize = 0;
    const VecCoord& vertices = this->getVertices();
    const VecCoord& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    const bool hasTangents = vtangents.size() && vbitangents.size();

    positionsBufferSize = (vertices.size()*sizeof(Vec3f));
    normalsBufferSize = (vnormals.size()*sizeof(Vec3f));

    if (m_tex || d_putOnlyTexCoords.getValue() || !textures.empty())
    {
        textureCoordsBufferSize = vtexcoords.size() * sizeof(vtexcoords[0]);

        if (hasTangents)
        {
            tangentsBufferSize = vtangents.size() * sizeof(vtangents[0]);
            bitangentsBufferSize = vbitangents.size() * sizeof(vbitangents[0]);
        }
    }

    const size_t totalSize = positionsBufferSize + normalsBufferSize + textureCoordsBufferSize +
            tangentsBufferSize + bitangentsBufferSize;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 long(totalSize),
                 nullptr,
                 GL_DYNAMIC_DRAW);

    updateVertexBuffer();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void OglModel::initEdgesIndicesBuffer()
{
    const Inherit::VecVisualEdge& edges = this->getEdges();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboEdges);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, long(edges.size()*sizeof(edges[0])), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    updateEdgesIndicesBuffer();
}

void OglModel::initTrianglesIndicesBuffer()
{
    const Inherit::VecVisualTriangle& triangles = this->getTriangles();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, long(triangles.size()*sizeof(triangles[0])), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    updateTrianglesIndicesBuffer();
}

void OglModel::initQuadsIndicesBuffer()
{
    const Inherit::VecVisualQuad& quads = this->getQuads();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, long(quads.size()*sizeof(quads[0])), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    updateQuadsIndicesBuffer();
}

void OglModel::updateVertexBuffer()
{

    const VecCoord& vertices = this->getVertices();
    const VecCoord& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    const bool hasTangents = vtangents.size() && vbitangents.size();

    size_t positionsBufferSize, normalsBufferSize;
    size_t textureCoordsBufferSize = 0, tangentsBufferSize = 0, bitangentsBufferSize = 0;

    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    normalsBufferSize = (vnormals.size()*sizeof(vnormals[0]));
    const void* positionBuffer = vertices.data();
    const void* normalBuffer = vnormals.data();

    // use only temporary float buffers if vertices/normals are using double
    if constexpr(std::is_same_v<Coord, sofa::type::Vec3d>)
    {
        verticesTmpBuffer.resize( vertices.size() );
        normalsTmpBuffer.resize( vnormals.size() );

        copyVector(vertices, verticesTmpBuffer);
        copyVector(vnormals, normalsTmpBuffer);

        positionBuffer = verticesTmpBuffer.data();
        normalBuffer = normalsTmpBuffer.data();
    }

    positionsBufferSize = (vertices.size()*sizeof(Vec3f));
    normalsBufferSize = (vnormals.size()*sizeof(Vec3f));

    if (m_tex || d_putOnlyTexCoords.getValue() || !textures.empty())
    {
        textureCoordsBufferSize = (vtexcoords.size() * sizeof(vtexcoords[0]));

        if (hasTangents)
        {
            tangentsBufferSize = (vtangents.size() * sizeof(vtangents[0]));
            bitangentsBufferSize = (vbitangents.size() * sizeof(vbitangents[0]));
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    //Positions
    glBufferSubData(GL_ARRAY_BUFFER,
        0,
        positionsBufferSize,
        positionBuffer);

    //Normals
    glBufferSubData(GL_ARRAY_BUFFER,
        positionsBufferSize,
        normalsBufferSize,
        normalBuffer);

    ////Texture coords
    if (m_tex || d_putOnlyTexCoords.getValue() || !textures.empty())
    {
        glBufferSubData(GL_ARRAY_BUFFER,
            positionsBufferSize + normalsBufferSize,
            textureCoordsBufferSize,
            vtexcoords.data());

        if (hasTangents)
        {
            glBufferSubData(GL_ARRAY_BUFFER,
                positionsBufferSize + normalsBufferSize + textureCoordsBufferSize,
                tangentsBufferSize,
                vtangents.data());

            glBufferSubData(GL_ARRAY_BUFFER,
                positionsBufferSize + normalsBufferSize + textureCoordsBufferSize + tangentsBufferSize,
                bitangentsBufferSize,
                vbitangents.data());
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void OglModel::updateEdgesIndicesBuffer()
{
    const VecVisualEdge& edges = this->getEdges();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboEdges);

    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, long(edges.size()*sizeof(edges[0])), &edges[0]);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateTrianglesIndicesBuffer()
{
    const VecVisualTriangle& triangles = this->getTriangles();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);

    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, long(triangles.size() * sizeof(triangles[0])), &triangles[0]);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateQuadsIndicesBuffer()
{
    const VecVisualQuad& quads = this->getQuads();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);

    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, long(quads.size() * sizeof(quads[0])), &quads[0]);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
void OglModel::updateBuffers()
{
    const Inherit::VecVisualEdge& edges = this->getEdges();
    const Inherit::VecVisualTriangle& triangles = this->getTriangles();
    const Inherit::VecVisualQuad& quads = this->getQuads();

    const VecCoord& vertices = this->getVertices();
    const VecDeriv& normals = this->getVnormals();
    const VecTexCoord& texCoords = this->getVtexcoords();
    const VecCoord& tangents = this->getVtangents();
    const VecCoord& bitangents = this->getVbitangents();

    if (initDone)
    {
        if(!VBOGenDone)
        {
            createVertexBuffer();
            //Index Buffer Object
            //Edges indices
            if(edges.size() > 0)
                createEdgesIndicesBuffer();
            //Triangles indices
            if(triangles.size() > 0)
                createTrianglesIndicesBuffer();
            //Quads indices
            if(quads.size() > 0)
                createQuadsIndicesBuffer();
        }
        //Update VBO & IBO
        else
        {
            // if any topology change then resize buffer
            if (oldVerticesSize != vertices.size() ||
                oldNormalsSize != normals.size() ||
                oldTexCoordsSize != texCoords.size() ||
                oldTangentsSize != tangents.size() ||
                oldBitangentsSize != bitangents.size())
            {
                initVertexBuffer();
            }
            else
            {
                // if no topology change but vertices changes then update buffer
                if (this->modified)
                {
                    updateVertexBuffer();
                }
            }


            //Indices
            //Edges
            if (useEdges && !edges.empty())
            {

                if(oldEdgesSize != edges.size())
                    initEdgesIndicesBuffer();
                else
                    if(edgesRevision < d_edges.getCounter())
                        updateEdgesIndicesBuffer();

            }
            else if (edges.size() > 0)
                createEdgesIndicesBuffer();

            //Triangles
            if (useTriangles && !triangles.empty())
            {
                if (oldTrianglesSize != triangles.size())
                    initTrianglesIndicesBuffer();
                else
                    if (trianglesRevision < d_triangles.getCounter())
                        updateTrianglesIndicesBuffer();
            }
            else if (triangles.size() > 0)
                createTrianglesIndicesBuffer();

            //Quads
            if (useQuads && !quads.empty())
            {
                if(oldQuadsSize != quads.size())
                    initQuadsIndicesBuffer();
                else
                    if (quadsRevision < d_quads.getCounter())
                        updateQuadsIndicesBuffer();
            }
            else if (quads.size() > 0)
                createQuadsIndicesBuffer();
        }

        oldVerticesSize = vertices.size();
        oldNormalsSize = normals.size();
        oldTexCoordsSize = texCoords.size();
        oldTangentsSize = tangents.size();
        oldBitangentsSize = bitangents.size();
        oldEdgesSize = edges.size();
        oldTrianglesSize = triangles.size();
        oldQuadsSize = quads.size();

        edgesRevision = d_edges.getCounter();
        trianglesRevision = d_triangles.getCounter();
        quadsRevision = d_quads.getCounter();
    }
}


GLenum OglModel::getGLenum(const char* c ) const
{

    if ( strcmp( c, "GL_ZERO") == 0)
    {
        return GL_ZERO;
    }
    else if  ( strcmp( c, "GL_ONE") == 0)
    {
        return GL_ONE;
    }
    else if (strcmp( c, "GL_SRC_ALPHA") == 0 )
    {
        return GL_SRC_ALPHA;
    }
    else if (strcmp( c, "GL_ONE_MINUS_SRC_ALPHA") == 0 )
    {
        return GL_ONE_MINUS_SRC_ALPHA;
    }
    // .... add other OGL symbolic constants
    // glBlendEquation Value
    else if  ( strcmp( c, "GL_FUNC_ADD") == 0)
    {
        return GL_FUNC_ADD;
    }
    else if (strcmp( c, "GL_FUNC_SUBTRACT") == 0 )
    {
        return GL_FUNC_SUBTRACT;
    }
    else if (strcmp( c, "GL_MAX") == 0 )
    {
        return GL_MAX;
    }
    else if (strcmp( c, "GL_MIN") == 0 )
    {
        return GL_MIN;
    }
    else
    {
        msg_warning() << " OglModel - not valid or not supported openGL enum value: " << c ;
        return GL_ZERO;
    }
}


} // namespace sofa::gl::component::rendering3d
