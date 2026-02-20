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
#include <cassert>
#include <sofa/gl/FrameBufferObject.h>
#include <sofa/helper/logging/Messaging.h>


namespace sofa::gl
{

FrameBufferObject::FrameBufferObject(bool depthTexture, bool enableDepth, bool enableColor, bool enableMipMap, GLint defaultWindowFramebuffer)
    :m_defaultWindowFramebufferID(defaultWindowFramebuffer)
    ,width(0)
    ,height(0)
    ,depthTextureID(0)
    ,colorTextureID(0)
    ,initialized(false)
    ,depthTexture(depthTexture)
    ,enableDepth(enableDepth)
    ,enableColor(enableColor)
    ,enableMipMap(enableMipMap)
{
}

FrameBufferObject::FrameBufferObject(const fboParameters& fboParams, bool depthTexture, bool enableDepth, bool enableColor, bool enableMipMap, GLint defaultWindowFramebuffer)
    :m_defaultWindowFramebufferID(defaultWindowFramebuffer)
    ,width(0)
    ,height(0)
    ,depthTextureID(0)
    ,colorTextureID(0)
    ,initialized(false)
    ,_fboParams(fboParams)
    ,depthTexture(depthTexture)
    ,enableDepth(enableDepth)
    ,enableColor(enableColor)
    ,enableMipMap(enableMipMap)
{
}


FrameBufferObject::~FrameBufferObject()
{
    destroy();
}

GLint FrameBufferObject::getCurrentFramebufferID()
{
    GLint windowId;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &windowId);

    return windowId;
}

void FrameBufferObject::destroy()
{
    if(initialized)
    {
        if(enableDepth)
        {
            if(depthTexture)
                glDeleteTextures( 1, &depthTextureID );
            else
                glDeleteRenderbuffers(1, &depthTextureID);
        }

        if(enableColor)
        {
            glDeleteTextures( 1, &colorTextureID );
        }

        glDeleteFramebuffers( 1, &id );
        initialized = false;
    }
}

bool FrameBufferObject::checkFBO()
{
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch(status)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        return true;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        msg_error("FrameBufferObject") << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
        return false;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        msg_error("FrameBufferObject") << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
        return false;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        msg_error("FrameBufferObject") << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
        return false;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        msg_error("FrameBufferObject") << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
        return false;
    case GL_FRAMEBUFFER_UNSUPPORTED:
        msg_error("FrameBufferObject") << "GL_FRAMEBUFFER_UNSUPPORTED";
        return false;
    default:
        msg_error("FrameBufferObject") << "Unknown framebuffer error: " << status;
        return false;
    }
}

void FrameBufferObject::init(unsigned int width, unsigned height)
{
    if (!initialized)
    {
        this->width = width;
        this->height = height;
        glGenFramebuffers(1, &id);
        glBindFramebuffer(GL_FRAMEBUFFER, id);

        if(enableDepth)
        {
            createDepthBuffer();
            initDepthBuffer();

            //choice between rendering depth into a texture or a renderbuffer
            if(depthTexture)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureID, 0);
            else
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthTextureID);
        }

        if(enableColor)
        {
            createColorBuffer();
            initColorBuffer();
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTextureID, 0);
        }

        if (!checkFBO())
        {
            msg_error("FrameBufferObject") << "FBO is not complete after init ("
                << width << "x" << height << "), id=" << id
                << ", colorTex=" << colorTextureID << ", depthRB=" << depthTextureID;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, m_defaultWindowFramebufferID);

        if(enableColor)
        {
            glDrawBuffer(GL_BACK);
            glReadBuffer(GL_BACK);
        }

        initialized=true;
    }
    else
        setSize(width, height);
}


void FrameBufferObject::reinit(unsigned int width, unsigned height, bool lDepthTexture, bool lEnableDepth, bool lEnableColor )
{
    destroy();

    depthTexture = lDepthTexture;
    enableDepth = lEnableDepth;
    enableColor = lEnableColor;

    init(width, height);

}


void FrameBufferObject::start()
{
    if (initialized)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, id);

        if(enableColor)
        {
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glDrawBuffer(GL_COLOR_ATTACHMENT0);
        }
    }
}

void FrameBufferObject::stop()
{
    if (initialized)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, m_defaultWindowFramebufferID);

        if(enableColor)
        {
            glDrawBuffer(GL_BACK);
            glReadBuffer(GL_BACK);
        }
    }
}

GLuint FrameBufferObject::getID()
{
    return id;
}

GLuint FrameBufferObject::getDepthTexture()
{
    return depthTextureID;
}

GLuint FrameBufferObject::getColorTexture()
{
    return colorTextureID;
}

void FrameBufferObject::setSize(unsigned int width, unsigned height)
{
    if (initialized && width > 0 && height > 0)
    {
        this->width = width;
        this->height = height;

        if(enableDepth)
            initDepthBuffer();
        if(enableColor)
            initColorBuffer();
    }
}

void FrameBufferObject::createDepthBuffer()
{
    if(depthTexture)
        glGenTextures(1, &depthTextureID);
    else
        glGenRenderbuffers(1, &depthTextureID);
}

void FrameBufferObject::createColorBuffer()
{
    glGenTextures(1, &colorTextureID);
}

void FrameBufferObject::initDepthBuffer()
{
    if(depthTexture)
    {
        glBindTexture(GL_TEXTURE_2D, depthTextureID);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

        glTexImage2D(GL_TEXTURE_2D, 0, _fboParams.depthInternalformat , width, height, 0,GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    else
    {
        glBindRenderbuffer(GL_RENDERBUFFER, depthTextureID);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

    }
}

void FrameBufferObject::initColorBuffer()
{
    glBindTexture(GL_TEXTURE_2D, colorTextureID);
    if(enableMipMap)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    else
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

    glTexImage2D(GL_TEXTURE_2D, 0, _fboParams.colorInternalformat,  width, height, 0, _fboParams.colorFormat, _fboParams.colorType, nullptr);
    if(enableMipMap)
        glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

}


} // namespace sofa::gl
