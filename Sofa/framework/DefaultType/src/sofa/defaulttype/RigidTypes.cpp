/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*            (c) 2006-2021 MGH, INRIA, USTL, UJF, CNRS, InSimo                *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_DEFAULTTYPE_RIGIDTYPES_CPP

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraint.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixGeneric.inl>

namespace sofa::linearalgebra
{

template class SOFA_DEFAULTTYPE_API CompressedRowSparseMatrixConstraint<defaulttype::Rigid2dTypes::Deriv>;
template class SOFA_DEFAULTTYPE_API CompressedRowSparseMatrixConstraint<defaulttype::Rigid3dTypes::Deriv>;

} // namespace sofa::linearalgebra
