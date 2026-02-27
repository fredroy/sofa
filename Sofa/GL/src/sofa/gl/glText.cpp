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
#include <sofa/gl/glText.inl>

namespace sofa::gl
{
using namespace sofa::type;
using std::string;

SOFA_GL_API const std::string GlText::ASCII_TEXTURE_PATH("textures/texture_ascii_smooth.png");
SOFA_GL_API sofa::helper::io::Image *GlText::s_asciiImage = nullptr;
SOFA_GL_API sofa::gl::Texture* GlText::s_asciiTexture = nullptr;

void GlText::initTexture()
{
    if (s_asciiImage == nullptr)
    {
        s_asciiImage = helper::io::Image::Create(ASCII_TEXTURE_PATH);
    }
    if (s_asciiTexture == nullptr && s_asciiImage != nullptr)
    {
        s_asciiTexture = new sofa::gl::Texture(s_asciiImage, false, true, false );
    }
}

GlText::GlText()
{
}

GlText::GlText ( const string& text )
{
    this->text = text;
}

GlText::GlText ( const string& text, const type::Vec3& position )
{
    this->text = text;
    this->position = position;
}

GlText::GlText ( const string& text, const type::Vec3& position, const double& scale )
{
    this->text = text;
    this->position = position;
    this->scale = scale;
}

GlText::~GlText()
{
}


void GlText::setText ( const string& text )
{
    this->text = text;
}

void GlText::update ( const type::Vec3& position )
{
    this->position = position;
}

void GlText::update ( const double& scale )
{
    this->scale = scale;
}


void GlText::draw() {}

void GlText::textureDraw_Overlay(const char*, const double) {}

void GlText::textureDraw_Indices(const type::vector<type::Vec3>&, const float&) {}

} // namespace sofa::gl
