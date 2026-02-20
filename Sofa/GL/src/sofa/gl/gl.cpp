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
#include <sofa/gl/gl.h>
#include <cstdio>

/// Returns the GL major version (cached after first call)
static int getGLMajorVersion()
{
    static int major = -1;
    if (major < 0)
    {
        major = 0;
        const char* v = reinterpret_cast<const char*>(glGetString(GL_VERSION));
        if (v) std::sscanf(v, "%d", &major);
    }
    return major;
}

SOFA_GL_API const char* GetGlExtensionsList()
{
    return reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
}


SOFA_GL_API bool CanUseGlExtension(const std::string& ext)
{
    // In core profile, ARB extensions that were promoted to core are no longer
    // listed in the extension string. Check GL version for known promotions.
    const int glMajor = getGLMajorVersion();
    if (glMajor >= 2)
    {
        // These extensions became core in GL 1.5 - 2.0
        if (ext == "GL_ARB_vertex_buffer_object" ||
            ext == "GL_ARB_shader_objects" ||
            ext == "GL_ARB_shading_language_100" ||
            ext == "GL_ARB_vertex_shader" ||
            ext == "GL_ARB_fragment_shader" ||
            ext == "GL_ARB_multitexture")
            return true;
    }

    // Try legacy glGetString(GL_EXTENSIONS) first (works in compatibility profile)
    const char * extensions = GetGlExtensionsList();
    if( extensions && std::string(extensions).find( ext ) != std::string::npos )
        return true;

    // Core profile: use indexed extension query (glGetString(GL_EXTENSIONS) is invalid)
    GLint numExtensions = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
    for (GLint i = 0; i < numExtensions; ++i)
    {
        const char* e = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));
        if (e && ext == e)
            return true;
    }
    return false;
}
