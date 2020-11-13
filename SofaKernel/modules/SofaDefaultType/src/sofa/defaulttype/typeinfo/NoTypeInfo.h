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
#include <string>

#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfoDynamicWrapper.h>

namespace sofa::defaulttype
{

class StaticNoTypeInfo : public IncompleteTypeInfo<StaticNoTypeInfo>
{
public:
    static std::string name(){ return "NoTypeInfo"; }
};

/** *******************************************************************************
 * @brief A unique singleton describe a totally missing typeinfo.
 **********************************************************************************/
class NoTypeInfo : public DataTypeInfoDynamicWrapper<StaticNoTypeInfo>
{
    public:
        static const NoTypeInfo* get(){ static NoTypeInfo none; return &none; }
};

} /// namespace sofa::defaulttype
