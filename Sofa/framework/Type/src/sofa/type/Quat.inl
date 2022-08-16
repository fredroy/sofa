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

#include <sofa/type/Quat.h>
#include <iostream>

namespace sofa::type
{

/// write to an output stream
template<class Real> std::ostream& operator << (std::ostream& out, const Quat<Real>& v)
{
    out << v[0] << " " << v[1] << " " << v[2] << " " << v[3];
    return out;
}

/// read from an input stream
template<class Real> std::istream& operator >> (std::istream& in, Quat<Real>& v)
{
    in >> v[0] >> v[1] >> v[2] >> v[3];
    return in;
}

} // namespace sofa::type
