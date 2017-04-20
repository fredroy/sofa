/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/helper/gl/Utilities.inl>

namespace sofa
{

namespace helper
{

namespace gl
{

template SOFA_HELPER_API
int Utilities::glhUnProject<float>(float winx, float winy, float winz,
    float *modelview, float *projection, const int *viewport,
    float *objectCoordinate);

template SOFA_HELPER_API
int Utilities::glhUnProject<double>(double winx, double winy, double winz,
    double *modelview, double *projection, const int *viewport,
    double *objectCoordinate);

template SOFA_HELPER_API
void Utilities::computeNormalMatrix<float>(float* modelviewMatrix, float* normalMatrix);

template SOFA_HELPER_API
void Utilities::computeNormalMatrix<double>(double* modelviewMatrix, double* normalMatrix);

} // namespace gl

} // namespace helper

} // namespace sofa

