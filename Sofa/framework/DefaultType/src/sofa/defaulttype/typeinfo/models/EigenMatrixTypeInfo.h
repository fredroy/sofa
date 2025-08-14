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

#include <sofa/type/Mat.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>

namespace sofa::defaulttype
{

template<class TDataType>
struct EigenMatrixTypeInfo
{
    using EigenMatrixDataType = TDataType;
    using DataType = TDataType;

    typedef typename EigenMatrixDataType::value_type BaseType;
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = BaseTypeInfo::ValidInfo       };
    enum { FixedSize       = BaseTypeInfo::FixedSize       };
    enum { ZeroConstructor = BaseTypeInfo::ZeroConstructor };
    enum { SimpleCopy      = BaseTypeInfo::SimpleCopy      };
    enum { SimpleLayout    = BaseTypeInfo::SimpleLayout    };
    enum { Integer         = BaseTypeInfo::Integer         };
    enum { Scalar          = BaseTypeInfo::Scalar          };
    enum { Text            = BaseTypeInfo::Text            };
    enum { CopyOnWrite     = 1                             };
    enum { Container       = 1                             };

    enum { Size = EigenMatrixDataType::static_size };

    static sofa::Size size()
    {
        return Size;
    }

    static sofa::Size byteSize()
    {
        return ValueTypeInfo::byteSize();
    }

    static sofa::Size size(const EigenMatrixDataType& )
    {
        return size();
    }

    static bool setSize(EigenMatrixDataType& , sofa::Size )
    {
        return false;
    }

    template <typename T>
    static void getValue(const EigenMatrixDataType &data, sofa::Size index, T& value)
    {
        int l = index%data.cols();
        int c = index/data.cols();
        value = data(l,c);
    }

    template<typename T>
    static void setValue(EigenMatrixDataType &data, sofa::Size index, const T& value )
    {
        int l = index%data.cols();
        int c = index/data.cols();
        data(l,c) = value;
    }

    static void getValueString(const EigenMatrixDataType &data, sofa::Size index, std::string& value)
    {
        int l = index%data.cols();
        int c = index/data.cols();
        BaseTypeInfo::getValueString(data(l,c), index, value);
    }

    static void setValueString(EigenMatrixDataType &data, sofa::Size index, const std::string& value )
    {
        int l = index%data.cols();
        int c = index/data.cols();
        BaseTypeInfo::setValueString(data(l,c), index, value);
    }

    static const void* getValuePtr(const EigenMatrixDataType& data)
    {
        return data.ptr();
    }

    static void* getValuePtr(EigenMatrixDataType& data)
    {
        return data.ptr();
    }
};

} /// namespace sofa::defaulttype

