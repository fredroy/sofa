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
#define SOFA_COMPONENT_VISUAL_VISUALMODELIMPL_CPP
#include <sofa/component/visual/VisualModelImpl.inl>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/topology/TopologyData.inl>

namespace sofa::component::visual
{

using namespace sofa::defaulttype;

int TVisualModelImplClass = core::RegisterObject("Generic visual model. If a viewer is active it will replace the VisualModel alias, otherwise nothing will be displayed.")
        .add< TVisualModelImpl<defaulttype::Vec3Types>>()
        .addAlias("VisualModelImpl") 
        .addAlias("VisualModel")
        ;

template class SOFA_COMPONENT_VISUAL_API TVisualModelImpl<defaulttype::Vec3Types>;


} // namespace sofa::component::visual
