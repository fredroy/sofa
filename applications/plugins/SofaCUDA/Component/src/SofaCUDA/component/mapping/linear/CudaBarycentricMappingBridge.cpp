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
#include <SofaCUDA/component/config.h>
#include <SofaCUDA/component/mapping/linear/CudaBarycentricMappingBridge.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mapping::linear
{

using namespace sofa::defaulttype;

template class SOFACUDA_COMPONENT_API CudaBarycentricMappingBridge<Vec3Types, Vec3Types>;

} // namespace sofa::component::mapping::linear

namespace sofa::gpu::cuda
{

using namespace sofa::defaulttype;
using namespace sofa::component::mapping::linear;

void registerCudaBarycentricMappingBridge(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "GPU-accelerated BarycentricMapping bridge for CPU mechanical objects. "
        "Transfers data to GPU for computation, then transfers results back. "
        "Useful for large numbers of mapped points with standard Vec3 mechanical states.")
    .add< CudaBarycentricMappingBridge<Vec3Types, Vec3Types> >());
}

} // namespace sofa::gpu::cuda
