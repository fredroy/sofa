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

#include <SofaCUDA/component/config.h>
#include <sofa/core/Mapping.h>
#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/component/mapping/linear/CudaBarycentricMapping.h>

namespace sofa::component::mapping::linear
{

/// BarycentricMapping between standard CPU mechanical objects that internally
/// accelerates computation on the GPU via CUDA.
///
/// This component is useful when the source and mapped mechanical objects use
/// standard CPU vector types (e.g. Vec3Types) but the mapping computation
/// (apply, applyJ, applyJT) should run on the GPU for performance — typically
/// when the number of mapped points is very large.
///
/// Internally, positions/forces are transferred to GPU memory (with double-to-float
/// conversion if needed), the existing CUDA barycentric mapping kernels are executed,
/// and results are transferred back to CPU memory.
template <class TIn, class TOut>
class CudaBarycentricMappingBridge : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CudaBarycentricMappingBridge, TIn, TOut),
               SOFA_TEMPLATE2(core::Mapping, TIn, TOut));

    using In = TIn;
    using Out = TOut;
    using InVecCoord = typename In::VecCoord;
    using InVecDeriv = typename In::VecDeriv;
    using OutVecCoord = typename Out::VecCoord;
    using OutVecDeriv = typename Out::VecDeriv;

protected:
    using GpuTypes = gpu::cuda::CudaVec3fTypes;
    using GpuMapper = BarycentricMapperMeshTopology<GpuTypes, GpuTypes>;
    using GpuVec = typename GpuTypes::VecCoord; // CudaVector<Vec3f>

    std::unique_ptr<GpuMapper> m_gpuMapper;

    GpuVec m_gpuIn;   ///< GPU buffer for source-side data
    GpuVec m_gpuOut;  ///< GPU buffer for mapped-side data

    CudaBarycentricMappingBridge();

public:
    ~CudaBarycentricMappingBridge() override = default;

    void init() override;

    void apply(const core::MechanicalParams* mparams,
               Data<OutVecCoord>& out,
               const Data<InVecCoord>& in) override;

    void applyJ(const core::MechanicalParams* mparams,
                Data<OutVecDeriv>& out,
                const Data<InVecDeriv>& in) override;

    void applyJT(const core::MechanicalParams* mparams,
                 Data<InVecDeriv>& out,
                 const Data<OutVecDeriv>& in) override;

    void applyJT(const core::ConstraintParams* cparams,
                 Data<typename In::MatrixDeriv>& out,
                 const Data<typename Out::MatrixDeriv>& in) override;
};

} // namespace sofa::component::mapping::linear
