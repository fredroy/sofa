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

#include <SofaCUDA/component/visual/CudaVisualModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/gl/template.h>

namespace sofa
{


namespace gpu::cuda
{

extern "C"
{
    void CudaVisualModelCuda3f_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);
    void CudaVisualModelCuda3f_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex, const void* triangles, const void* quads, void* vnormals, const void* x);

    void CudaVisualModelCuda3f1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);
    void CudaVisualModelCuda3f1_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex, const void* triangles, const void* quads, void* vnormals, const void* x);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void CudaVisualModelCuda3d_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);
    void CudaVisualModelCuda3d_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex, const void* triangles, const void* quads, void* vnormals, const void* x);

    void CudaVisualModelCuda3d1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);
    void CudaVisualModelCuda3d1_calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex, const void* triangles, const void* quads, void* vnormals, const void* x);

#endif // SOFA_GPU_CUDA_DOUBLE

} // extern "C"

template<>
class CudaKernelsCudaVisualModel<CudaVec3fTypes>
{
public:
    static void calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3f_calcTNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3f_calcQNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
    {   CudaVisualModelCuda3f_calcVNormals(nbElem, nbVertex, nbElemPerVertex, velems, vnormals, fnormals, x); }
    static void calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex, const void* triangles, const void* quads, void* vnormals, const void* x)
    {   CudaVisualModelCuda3f_calcNormalsAtomic(nbTriangles, nbQuads, nbVertex, triangles, quads, vnormals, x); }
};

template<>
class CudaKernelsCudaVisualModel<CudaVec3f1Types>
{
public:
    static void calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3f1_calcTNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3f1_calcQNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
    {   CudaVisualModelCuda3f1_calcVNormals(nbElem, nbVertex, nbElemPerVertex, velems, vnormals, fnormals, x); }
    static void calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex, const void* triangles, const void* quads, void* vnormals, const void* x)
    {   CudaVisualModelCuda3f1_calcNormalsAtomic(nbTriangles, nbQuads, nbVertex, triangles, quads, vnormals, x); }
};

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
class CudaKernelsCudaVisualModel<CudaVec3dTypes>
{
public:
    static void calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3d_calcTNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3d_calcQNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
    {   CudaVisualModelCuda3d_calcVNormals(nbElem, nbVertex, nbElemPerVertex, velems, vnormals, fnormals, x); }
    static void calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex, const void* triangles, const void* quads, void* vnormals, const void* x)
    {   CudaVisualModelCuda3d_calcNormalsAtomic(nbTriangles, nbQuads, nbVertex, triangles, quads, vnormals, x); }
};

template<>
class CudaKernelsCudaVisualModel<CudaVec3d1Types>
{
public:
    static void calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3d1_calcTNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3d1_calcQNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
    {   CudaVisualModelCuda3d1_calcVNormals(nbElem, nbVertex, nbElemPerVertex, velems, vnormals, fnormals, x); }
    static void calcNormalsAtomic(unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbVertex, const void* triangles, const void* quads, void* vnormals, const void* x)
    {   CudaVisualModelCuda3d1_calcNormalsAtomic(nbTriangles, nbQuads, nbVertex, triangles, quads, vnormals, x); }
};

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace gpu::cuda


namespace component::visualmodel
{

using namespace gpu::cuda;

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::init()
{
    Inherit1::init();
    Inherit2::init();

    if (!l_topology)
    {
        l_topology = this->getContext()->getMeshTopology();
    }

    // Initialize positions from topology if available
    if (l_topology && this->m_positions.getValue().empty())
    {
        const auto nbPoints = l_topology->getNbPoints();
        if (nbPoints > 0)
        {
            this->resize(nbPoints);
        }
    }

    updateTopologyAndNormals();
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::reinit()
{
    updateTopologyAndNormals();
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::handleTopologyChange()
{
    if (!l_topology) return;

    std::list<const core::topology::TopologyChange *>::const_iterator itBegin = l_topology->beginChange();
    const std::list<const core::topology::TopologyChange *>::const_iterator itEnd = l_topology->endChange();

    while (itBegin != itEnd)
    {
        const core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

        switch (changeType)
        {
        case core::topology::TRIANGLESREMOVED:
        {
            needUpdateTopology = true;
            break;
        }

        case core::topology::QUADSADDED:
        {
            needUpdateTopology = true;
            break;
        }

        case core::topology::QUADSREMOVED:
        {
            needUpdateTopology = true;
            break;
        }
        default:
            break;
        }
        ++itBegin;
    }
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::updateTopology()
{
    if (!l_topology) return;
    if (!needUpdateTopology) return;
    needUpdateTopology = false;

    {
        const SeqTriangles& t = l_topology->getTriangles();
        triangles.clear();
        if (!t.empty())
        {
            triangles.fastResize(t.size());
            std::copy(t.begin(), t.end(), triangles.hostWrite());
        }
    }
    {
        const SeqQuads& q = l_topology->getQuads();
        quads.clear();
        if (!q.empty())
        {
            quads.fastResize(q.size());
            std::copy(q.begin(), q.end(), quads.hostWrite());
        }
    }

    msg_info() << "CUDA CudaVisualModel: " << triangles.size() << " triangles, "
               << quads.size() << " quads, " << this->getSize() << " vertices.";
}


template<class TDataTypes>
void CudaVisualModel<TDataTypes>::updateNormals()
{
    if (!l_topology || this->getSize() == 0) return;

    // Access positions - we only read, so use getValue() and const_cast for deviceRead()
    // deviceRead() doesn't modify data, it just ensures GPU copy is valid
    const VecCoord& xConst = this->m_positions.getValue();
    VecCoord& x = const_cast<VecCoord&>(xConst);

    // Resize vertex normals - use fastResize to avoid unnecessary initialization
    VecDeriv& vnormals = *this->m_vnormals.beginEdit();
    vnormals.fastResize(x.size());

    // Use atomic-based normal computation: simpler, no velems/fnormals needed
    Kernels::calcNormalsAtomic(
        triangles.size(),
        quads.size(),
        x.size(),
        triangles.size() > 0 ? triangles.deviceRead() : nullptr,
        quads.size() > 0 ? quads.deviceRead() : nullptr,
        vnormals.deviceWrite(),
        x.deviceRead());

    this->m_vnormals.endEdit();
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::updateTopologyAndNormals()
{
    updateTopology();
    if (computeNormals.getValue())
        updateNormals();
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::doUpdateVisual(const core::visual::VisualParams* vparams)
{
    SOFA_UNUSED(vparams);
    // Only update if topology changed or positions were modified (e.g., by a mapping)
    if (needUpdateTopology || this->modified)
    {
        updateTopologyAndNormals();
        this->modified = false;
    }
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::doDrawVisual(const core::visual::VisualParams* vparams)
{
    const bool transparent = (matDiffuse.getValue()[3] < 1.0);
    if (!transparent) internalDraw(vparams);
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::drawTransparent(const core::visual::VisualParams* vparams)
{
    const bool transparent = (matDiffuse.getValue()[3] < 1.0);
    if (transparent) internalDraw(vparams);
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::drawShadow(const core::visual::VisualParams* vparams)
{
    const bool transparent = (matDiffuse.getValue()[3] < 1.0);
    if (!transparent /* && getCastShadow() */) internalDraw(vparams);
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::internalDraw(const core::visual::VisualParams* vparams)
{
#if SOFACUDA_CORE_HAVE_SOFA_GL == 1
    if (!vparams->displayFlags().getShowVisualModels()) return;
    if (!l_topology || this->getSize() == 0) return;

    const bool wireframe = vparams->displayFlags().getShowWireFrame();
    if (wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glEnable(GL_LIGHTING);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

    const type::Vec4f& diffuse = matDiffuse.getValue();
    const bool transparent = (diffuse[3] < 1.0f);

    // Set material properties
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, matAmbient.getValue().ptr());
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse.ptr());
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, matSpecular.getValue().ptr());
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, matEmissive.getValue().ptr());
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, matShininess.getValue());

    if (transparent)
    {
        glEnable(GL_BLEND);
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    const bool vbo = useVBO.getValue();
    const bool useNormals = computeNormals.getValue();

    // Access data without triggering CPU sync
    const VecCoord& xConst = this->m_positions.getValue();
    VecCoord& x = const_cast<VecCoord&>(xConst);

    // Determine GL type once
    constexpr GLenum glType = (sizeof(Real) == sizeof(double)) ? GL_DOUBLE : GL_FLOAT;

    // Set up vertex array - bufferRead syncs GPU data to GL buffer
    const GLuint vbo_x = vbo ? x.bufferRead(true) : 0;
    if (vbo_x)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
        glVertexPointer(3, glType, sizeof(Coord), nullptr);
    }
    else
    {
        glVertexPointer(3, glType, sizeof(Coord), x.hostRead());
    }

    // Set up normal array
    if (useNormals)
    {
        const VecDeriv& vnormalsConst = this->m_vnormals.getValue();
        VecDeriv& vnormals = const_cast<VecDeriv&>(vnormalsConst);
        const GLuint vbo_n = vbo ? vnormals.bufferRead(true) : 0;
        if (vbo_n)
        {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_n);
            glNormalPointer(glType, sizeof(Deriv), nullptr);
        }
        else
        {
            glNormalPointer(glType, sizeof(Deriv), vnormals.hostRead());
        }
        glEnableClientState(GL_NORMAL_ARRAY);
    }
    glEnableClientState(GL_VERTEX_ARRAY);

    // Draw triangles
    if (triangles.size() > 0)
    {
        const GLuint vbo_t = vbo ? triangles.bufferRead(true) : 0;
        if (vbo_t)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_t);
            glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, nullptr);
        }
        else
        {
            glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, triangles.hostRead());
        }
    }

    // Draw quads
    if (quads.size() > 0)
    {
        const GLuint vbo_q = vbo ? quads.bufferRead(true) : 0;
        if (vbo_q)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_q);
            glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, nullptr);
        }
        else
        {
            glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, quads.hostRead());
        }
    }

    // Cleanup state
    if (vbo)
    {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_LIGHTING);

    if (transparent)
    {
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }

    if (wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (vparams->displayFlags().getShowNormals())
    {
        // ShowNormals requires CPU access - debug feature, sync acceptable
        const VecDeriv& vnormals = this->m_vnormals.getValue();
        glColor3f(1.0, 1.0, 1.0);
        for (unsigned int i = 0; i < x.size(); i++)
        {
            glBegin(GL_LINES);
            sofa::gl::glVertexT(x[i]);
            Coord p = x[i] + vnormals[i] * 0.01;
            sofa::gl::glVertexT(p);
            glEnd();
        }
    }

#endif // SOFACUDA_CORE_HAVE_SOFA_GL == 1
}

template<class TDataTypes>
void CudaVisualModel<TDataTypes>::computeBBox(const core::ExecParams* params, bool)
{
    SOFA_UNUSED(params);

    if (this->getSize() == 0)
        return;

    const VecCoord& x = this->m_positions.getValue();

    SReal minBBox[3] = {std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max()};
    SReal maxBBox[3] = {-std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max()};

    for (unsigned int i = 0; i < x.size(); i++)
    {
        const Coord& p = x[i];
        for (int c = 0; c < 3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    this->f_bbox.setValue(sofa::type::TBoundingBox<SReal>(minBBox, maxBBox));
}


} // namespace component::visualmodel


} // namespace sofa
