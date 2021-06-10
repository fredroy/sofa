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
#include <SofaGeneralEngine/config.h>

#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/vectorData.h>

namespace sofa::component::engine
{

/**
 * Apply a merge operation to combine several inputs
 */
template <class VecT>
class MergeVectors : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MergeVectors,VecT),core::DataEngine);
    typedef VecT VecValue;
    typedef typename VecValue::value_type Value;

protected:
    MergeVectors();

    ~MergeVectors() override;
public:
    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse( sofa::core::objectmodel::BaseObjectDescription* arg ) override;

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields( const std::map<std::string,std::string*>& str ) override;

    void init() override;

    void reinit() override;

    void doUpdate() override;

    /// Returns the sofa template name. By default the name of the c++ class signature is exposed...
    /// More details on the name customization infrastructure is in NameDecoder.h
    static const std::string GetCustomTemplateName()
    {
        return Data<Value>::templateName();
    }

    Data<unsigned int> f_nbInputs; ///< Number of input vectors
    helper::vectorData<VecValue> vf_inputs;
    Data<VecValue> f_output; ///< Output vector

};

#if  !defined(SOFA_COMPONENT_ENGINE_MERGEVECTORS_CPP)

extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< type::vector<int> >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< type::vector<bool> >;
//extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< type::vector<std::string> >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< type::vector<defaulttype::Vec2u> >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< type::vector<double> >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< type::vector<defaulttype::Vec2d> >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< type::vector<defaulttype::Vec3d> >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< type::vector<defaulttype::Vec4d> >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< defaulttype::Rigid2Types::VecCoord >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< defaulttype::Rigid2Types::VecDeriv >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< defaulttype::Rigid3Types::VecCoord >;
extern template class SOFA_SOFAGENERALENGINE_API MergeVectors< defaulttype::Rigid3Types::VecDeriv >;
 
#endif

} //namespace sofa::component::engine
