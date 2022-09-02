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
#include <sofa/linearalgebra/config.h>

#include <sofa/linearalgebra/BTDMatrix.h>

namespace sofa::linearalgebra
{

template<> const char* BTDMatrix<1, double>::Name() { return "BTDMatrix1d"; }
template<> const char* BTDMatrix<2, double>::Name() { return "BTDMatrix2d"; }
template<> const char* BTDMatrix<3, double>::Name() { return "BTDMatrix3d"; }
template<> const char* BTDMatrix<4, double>::Name() { return "BTDMatrix4d"; }
template<> const char* BTDMatrix<5, double>::Name() { return "BTDMatrix5d"; }
template<> const char* BTDMatrix<6, double>::Name() { return "BTDMatrix6d"; }

template<> const char* BTDMatrix<1, float>::Name() { return "BTDMatrix1f"; }
template<> const char* BTDMatrix<2, float>::Name() { return "BTDMatrix2f"; }
template<> const char* BTDMatrix<3, float>::Name() { return "BTDMatrix3f"; }
template<> const char* BTDMatrix<4, float>::Name() { return "BTDMatrix4f"; }
template<> const char* BTDMatrix<5, float>::Name() { return "BTDMatrix5f"; }
template<> const char* BTDMatrix<6, float>::Name() { return "BTDMatrix6f"; }

} // namespace sofa::linearalgebra