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
#ifndef SOFA_HELPER_GL_UTILITIES_H
#define SOFA_HELPER_GL_UTILITIES_H

#include <sofa/helper/helper.h>


namespace sofa
{

namespace helper
{

namespace gl
{

struct SOFA_HELPER_API Utilities
{
public:

    template <typename Real>
    static int glhUnProject(Real winx, Real winy, Real winz,
        Real *modelview, Real *projection, const int *viewport,
        Real *objectCoordinate);

    template <typename Real>
    static void computeNormalMatrix(Real* modelviewMatrix, Real* normalMatrix);
};

extern template
int Utilities::glhUnProject<float>(float winx, float winy, float winz,
    float *modelview, float *projection, const int *viewport,
    float *objectCoordinate);

extern template
int Utilities::glhUnProject<double>(double winx, double winy, double winz,
    double *modelview, double *projection, const int *viewport,
    double *objectCoordinate);

extern template
void Utilities::computeNormalMatrix<float>(float* modelviewMatrix, float* normalMatrix);

extern template
void Utilities::computeNormalMatrix<double>(double* modelviewMatrix, double* normalMatrix);

} // namespace gl

} // namespace helper

} // namespace sofa


#endif // SOFA_HELPER_GL_UTILITIES_H
