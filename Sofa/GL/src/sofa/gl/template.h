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
#include <sofa/gl/gl.h>
#include <sofa/gl/config.h>

namespace sofa::gl
{

template<int N> inline void glVertexNv(const float*) {}
template<int N> inline void glVertexNv(const double*) {}
template<class Coord> inline void glVertexT(const Coord&) {}
template<> inline void glVertexT<double>(const double&) {}
template<> inline void glVertexT<float>(const float&) {}

template<int N> inline void glTexCoordNv(const float*) {}
template<int N> inline void glTexCoordNv(const double*) {}
template<class Coord> inline void glTexCoordT(const Coord&) {}
template<> inline void glTexCoordT<double>(const double&) {}
template<> inline void glTexCoordT<float>(const float&) {}

template<int N> inline void glNormalNv(const float*) {}
template<int N> inline void glNormalNv(const double*) {}
template<class Coord> inline void glNormalT(const Coord&) {}
template<> inline void glNormalT<double>(const double&) {}
template<> inline void glNormalT<float>(const float&) {}

inline void glTranslate(const float&, const float&, const float&) {}
inline void glTranslate(const double&, const double&, const double&) {}
template<int N> inline void glTranslateNv(const float*) {}
template<int N> inline void glTranslateNv(const double*) {}
template<class Coord> inline void glTranslateT(const Coord&) {}
template<> inline void glTranslateT<double>(const double&) {}
template<> inline void glTranslateT<float>(const float&) {}

inline void glScale(const float&, const float&, const float&) {}
inline void glScale(const double&, const double&, const double&) {}
inline void glRotate(const float&, const float&, const float&, const float&) {}
inline void glRotate(const double&, const double&, const double&, const double&) {}
inline void glMultMatrix(const float*) {}
inline void glMultMatrix(const double*) {}

} // namespace sofa::gl
