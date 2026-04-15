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

#include <SofaCUDA/component/mapping/linear/CudaBarycentricMappingBridge.h>
#include <SofaCUDA/component/mapping/linear/CudaBarycentricMapping.inl>
#include <sofa/core/Mapping.inl>
#include <sofa/helper/accessor.h>

#include <cstring> // memset, memcpy
#include <type_traits>

namespace sofa::component::mapping::linear
{

namespace
{

/// Upload a CPU vector of Vec3 to a GPU CudaVector of Vec3f.
/// Uses memcpy when the element types match (float), otherwise converts element-wise.
template <class GpuVec, class CpuVec>
void toGpu(GpuVec& gpu, const CpuVec& cpu)
{
    gpu.fastResize(cpu.size());
    if constexpr (std::is_same_v<typename CpuVec::value_type, typename GpuVec::value_type>)
    {
        std::memcpy(gpu.hostWrite(), cpu.data(),
                    cpu.size() * sizeof(typename GpuVec::value_type));
    }
    else
    {
        auto* dst = gpu.hostWrite();
        for (std::size_t i = 0; i < cpu.size(); ++i)
            for (int d = 0; d < 3; ++d)
                dst[i][d] = static_cast<float>(cpu[i][d]);
    }
}

/// Download a GPU CudaVector of Vec3f to a CPU vector of Vec3, overwriting.
/// Uses memcpy when the element types match (float), otherwise converts element-wise.
template <class CpuVec, class GpuVec>
void fromGpu(CpuVec& cpu, const GpuVec& gpu)
{
    cpu.resize(gpu.size());
    if constexpr (std::is_same_v<typename CpuVec::value_type, typename GpuVec::value_type>)
    {
        std::memcpy(cpu.data(), gpu.hostRead(),
                    gpu.size() * sizeof(typename GpuVec::value_type));
    }
    else
    {
        const auto* src = gpu.hostRead();
        for (std::size_t i = 0; i < gpu.size(); ++i)
            for (int d = 0; d < 3; ++d)
                cpu[i][d] = src[i][d];
    }
}

/// Download a GPU CudaVector of Vec3f and accumulate (+=) into a CPU vector.
template <class CpuVec, class GpuVec>
void accumulateFromGpu(CpuVec& cpu, const GpuVec& gpu)
{
    const auto* src = gpu.hostRead();
    const std::size_t n = std::min(cpu.size(), gpu.size());
    for (std::size_t i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d)
            cpu[i][d] += src[i][d];
}

} // anonymous namespace


template <class TIn, class TOut>
CudaBarycentricMappingBridge<TIn, TOut>::CudaBarycentricMappingBridge()
    : core::Mapping<TIn, TOut>()
{
}

template <class TIn, class TOut>
void CudaBarycentricMappingBridge<TIn, TOut>::init()
{
    core::Mapping<TIn, TOut>::init();

    if (!this->fromModel || !this->toModel)
    {
        msg_error() << "Source or mapped model not found";
        return;
    }

    // Get topology from source model
    auto* topology = dynamic_cast<core::topology::BaseMeshTopology*>(
        this->fromModel->getContext()->getMeshTopology());

    if (!topology)
    {
        msg_error() << "No BaseMeshTopology found in source model context";
        return;
    }

    // Create the GPU mapper (uses BaseMeshTopology, handles all element types)
    m_gpuMapper = std::make_unique<GpuMapper>(topology, nullptr);

    // Upload CPU positions to GPU vectors for mapper initialization.
    // The mapper's init() only accesses host data (element search is CPU-side),
    // so the CudaVector host pointers are sufficient.
    const auto& outPos = this->toModel->read(core::vec_id::read_access::position)->getValue();
    const auto& inPos = this->fromModel->read(core::vec_id::read_access::position)->getValue();

    GpuVec gpuOutPos, gpuInPos;
    toGpu(gpuInPos, inPos);
    toGpu(gpuOutPos, outPos);

    m_gpuMapper->init(gpuOutPos, gpuInPos);

    // Pre-build the transpose map so applyJT doesn't pay for it on first call
    m_gpuMapper->ensureTransposeMap();

    msg_info() << "CudaBarycentricMappingBridge initialized: "
               << outPos.size() << " mapped points, " << inPos.size() << " source points";
}

template <class TIn, class TOut>
void CudaBarycentricMappingBridge<TIn, TOut>::apply(
    const core::MechanicalParams* mparams,
    Data<OutVecCoord>& dOut,
    const Data<InVecCoord>& dIn)
{
    SOFA_UNUSED(mparams);
    if (!m_gpuMapper) return;

    const auto& in = dIn.getValue();
    auto out = sofa::helper::getWriteOnlyAccessor(dOut);

    // CPU → GPU (double → float, or memcpy if float)
    toGpu(m_gpuIn, in);

    // GPU kernel: out = J * in (assignment)
    m_gpuMapper->apply(m_gpuOut, m_gpuIn);

    // GPU → CPU (float → double, or memcpy if float)
    fromGpu(out.wref(), m_gpuOut);
}

template <class TIn, class TOut>
void CudaBarycentricMappingBridge<TIn, TOut>::applyJ(
    const core::MechanicalParams* mparams,
    Data<OutVecDeriv>& dOut,
    const Data<InVecDeriv>& dIn)
{
    SOFA_UNUSED(mparams);
    if (!m_gpuMapper) return;

    const auto& in = dIn.getValue();
    auto out = sofa::helper::getWriteOnlyAccessor(dOut);

    toGpu(m_gpuIn, in);
    m_gpuMapper->applyJ(m_gpuOut, m_gpuIn);
    fromGpu(out.wref(), m_gpuOut);
}

template <class TIn, class TOut>
void CudaBarycentricMappingBridge<TIn, TOut>::applyJT(
    const core::MechanicalParams* mparams,
    Data<InVecDeriv>& dOut,
    const Data<OutVecDeriv>& dIn)
{
    SOFA_UNUSED(mparams);
    if (!m_gpuMapper) return;

    const std::size_t inSize = m_gpuMapper->getInputSize();
    const std::size_t maxNOut = m_gpuMapper->getMaxNOut();
    if (inSize == 0 || maxNOut == 0) return;

    const auto& in = dIn.getValue();
    auto out = sofa::helper::getWriteAccessor(dOut); // read-write: existing forces

    // Upload mapped-side forces to GPU
    toGpu(m_gpuOut, in);

    // Allocate output buffer. Use the assignment kernel (=) directly with the
    // transpose map, avoiding the need to zero-fill + sync + use the += kernel.
    m_gpuIn.fastResize(inSize);

    // Call the assignment kernel with the transpose map data.
    // MeshMapperCuda3f_apply uses = (not +=), so no zero-initialization needed.
    gpu::cuda::MeshMapperCuda3f_apply(
        static_cast<unsigned int>(inSize),
        static_cast<unsigned int>(maxNOut),
        m_gpuMapper->getTransposeMapData().deviceRead(),
        m_gpuIn.deviceWrite(),
        m_gpuOut.deviceRead());

    // Accumulate GPU result into CPU output
    accumulateFromGpu(out.wref(), m_gpuIn);
}

template <class TIn, class TOut>
void CudaBarycentricMappingBridge<TIn, TOut>::applyJT(
    const core::ConstraintParams* cparams,
    Data<typename In::MatrixDeriv>& out,
    const Data<typename Out::MatrixDeriv>& in)
{
    SOFA_UNUSED(cparams);
    SOFA_UNUSED(out);
    SOFA_UNUSED(in);
    // MatrixDeriv applyJT is not GPU-accelerated.
    // This path is used for constraint Jacobian transpose and is typically
    // sparse, making GPU acceleration unnecessary.
}

} // namespace sofa::component::mapping::linear
