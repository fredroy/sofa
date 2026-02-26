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

#include <SofaCUDA/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.inl>
#include <sofa/core/MechanicalParams.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

namespace
{

/// Expand a 10-bit integer to 30 bits by inserting 2 zero bits between each bit.
inline uint32_t expandBits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/// Compute a 30-bit Morton code for a 3D point in [0, 1]^3.
inline uint32_t morton3D(float x, float y, float z)
{
    x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
    y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
    z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
    return expandBits(static_cast<uint32_t>(x)) * 4
         + expandBits(static_cast<uint32_t>(y)) * 2
         + expandBits(static_cast<uint32_t>(z));
}

} // anonymous namespace

namespace sofa::gpu::cuda
{

extern "C"
{
void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, const void* elems, void* state, void* f, const void* x);
void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, const void* elems, const void* state, void* df, const void* dx, double kFactor);

void TetrahedronFEMForceFieldCuda3f_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
void TetrahedronFEMForceFieldCuda3f_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);

#ifdef SOFA_GPU_CUDA_DOUBLE

void TetrahedronFEMForceFieldCuda3d_addForce(unsigned int nbElem, const void* elems, void* state, void* f, const void* x);
void TetrahedronFEMForceFieldCuda3d_addDForce(unsigned int nbElem, const void* elems, const void* state, void* df, const void* dx, double kFactor);

void TetrahedronFEMForceFieldCuda3d_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
void TetrahedronFEMForceFieldCuda3d_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);

#endif // SOFA_GPU_CUDA_DOUBLE

} // extern "C"

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3fTypes>
{
public:
    static void addForce(unsigned int nbElem, const void* elems, void* state, void* f, const void* x)
    {   TetrahedronFEMForceFieldCuda3f_addForce(nbElem, elems, state, f, x); }
    static void addDForce(unsigned int nbElem, const void* elems, const void* state, void* df, const void* dx, double kFactor)
    {   TetrahedronFEMForceFieldCuda3f_addDForce(nbElem, elems, state, df, dx, kFactor); }

    static void getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
    {   TetrahedronFEMForceFieldCuda3f_getRotations(nbElem, nbVertex, initState, state, rotationIdx, rotations); }

    static void getRotationsElement(unsigned int nbElem, const void* rotationsAos, void* rotations)
    {   TetrahedronFEMForceFieldCuda3f_getElementRotations(nbElem, rotationsAos, rotations); }
};

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3dTypes>
{
public:
    static void addForce(unsigned int nbElem, const void* elems, void* state, void* f, const void* x)
    {   TetrahedronFEMForceFieldCuda3d_addForce(nbElem, elems, state, f, x); }
    static void addDForce(unsigned int nbElem, const void* elems, const void* state, void* df, const void* dx, double kFactor)
    {   TetrahedronFEMForceFieldCuda3d_addDForce(nbElem, elems, state, df, dx, kFactor); }

    static void getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
    {   TetrahedronFEMForceFieldCuda3d_getRotations(nbElem, nbVertex, initState, state, rotationIdx, rotations); }

    static void getRotationsElement(unsigned int nbElem, const void* rotationsAos, void* rotations)
    {   TetrahedronFEMForceFieldCuda3d_getElementRotations(nbElem, rotationsAos, rotations); }
};

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace sofa::gpu::cuda

namespace sofa::component::solidmechanics::fem::elastic
{

using namespace gpu::cuda;

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::reinit(Main* m)
{
    if (!m->l_topology->getTetrahedra().empty())
    {
        m->_indexedElements = & (m->l_topology->getTetrahedra());
    }

    Data& data = m->data;
    m->strainDisplacements.resize( m->_indexedElements->size() );
    m->materialsStiffnesses.resize(m->_indexedElements->size() );

    const VecElement& elems = *m->_indexedElements;

    const VecCoord& p = m->mstate->read(core::vec_id::read_access::restPosition)->getValue();
    m->d_initialPoints.setValue(p);

    m->rotations.resize( m->_indexedElements->size() );
    m->_initialRotations.resize( m->_indexedElements->size() );
    m->_rotationIdx.resize(m->_indexedElements->size() *4);
    m->_rotatedInitialElements.resize(m->_indexedElements->size());

    std::vector<int> activeElems;
    for (unsigned int i=0; i<elems.size(); i++)
    {
        {
            activeElems.push_back(i);
        }
    }

    for (unsigned int i=0; i<activeElems.size(); i++)
    {
        int ei = activeElems[i];
        Index a = elems[ei][0];
        Index b = elems[ei][1];
        Index c = elems[ei][2];
        Index d = elems[ei][3];
        m->computeMaterialStiffness(ei,a,b,c,d);
        m->initLarge(ei,a,b,c,d);
    }

    std::map<int,int> nelems;
    for (unsigned int i=0; i<activeElems.size(); i++)
    {
        int ei = activeElems[i];
        const Element& e = elems[ei];
        for (unsigned int j=0; j<e.size(); j++)
            ++nelems[e[j]];
    }
    int v0 = 0;
    int nbv = 0;
    if (!nelems.empty())
    {
        v0 = nelems.begin()->first;
        nbv = nelems.rbegin()->first - v0 + 1;
    }

    // Reorder elements by spatial locality using Morton/Z-order curve.
    // Spatially adjacent elements get consecutive GPU thread indices,
    // improving L2 cache hit rates for scattered position reads.
    {
        const unsigned int nActive = static_cast<unsigned int>(activeElems.size());
        std::vector<float> cx(nActive), cy(nActive), cz(nActive);
        float minx = std::numeric_limits<float>::max(), miny = minx, minz = minx;
        float maxx = std::numeric_limits<float>::lowest(), maxy = maxx, maxz = maxx;

        for (unsigned int i = 0; i < nActive; ++i)
        {
            const Element& el = elems[activeElems[i]];
            float x = 0, y = 0, z = 0;
            for (unsigned int j = 0; j < el.size(); ++j)
            {
                x += static_cast<float>(p[el[j]][0]);
                y += static_cast<float>(p[el[j]][1]);
                z += static_cast<float>(p[el[j]][2]);
            }
            cx[i] = x * 0.25f; cy[i] = y * 0.25f; cz[i] = z * 0.25f;
            minx = std::min(minx, cx[i]); miny = std::min(miny, cy[i]); minz = std::min(minz, cz[i]);
            maxx = std::max(maxx, cx[i]); maxy = std::max(maxy, cy[i]); maxz = std::max(maxz, cz[i]);
        }

        const float invx = (maxx > minx) ? 1.0f / (maxx - minx) : 0.0f;
        const float invy = (maxy > miny) ? 1.0f / (maxy - miny) : 0.0f;
        const float invz = (maxz > minz) ? 1.0f / (maxz - minz) : 0.0f;

        std::vector<uint32_t> mortonCodes(nActive);
        for (unsigned int i = 0; i < nActive; ++i)
            mortonCodes[i] = morton3D(
                (cx[i] - minx) * invx,
                (cy[i] - miny) * invy,
                (cz[i] - minz) * invz);

        std::vector<unsigned int> order(nActive);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
            [&mortonCodes](unsigned int a, unsigned int b) {
                return mortonCodes[a] < mortonCodes[b];
            });

        std::vector<int> sorted(nActive);
        for (unsigned int i = 0; i < nActive; ++i)
            sorted[i] = activeElems[order[i]];
        activeElems = std::move(sorted);
    }

    data.init(activeElems.size(), v0, nbv);
    data.elemReorder = activeElems;

    for (unsigned eindex = 0; eindex < activeElems.size(); ++eindex)
    {
        int ei = activeElems[eindex];
        const Element& e = elems[ei];

        const Coord& a = m->_rotatedInitialElements[ei][0];
        const Coord& b = m->_rotatedInitialElements[ei][1];
        const Coord& c = m->_rotatedInitialElements[ei][2];
        const Coord& d = m->_rotatedInitialElements[ei][3];
        data.setE(eindex, e, a, b, c, d, m->materialsStiffnesses[ei], m->strainDisplacements[ei]);
    }
}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addForce(Main* m, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/)
{
    if (m->needUpdateTopology)
    {
        reinit(m);
        m->needUpdateTopology = false;
    }
    Data& data = m->data;

    f.resize(x.size());

    Kernels::addForce(
                data.size(),
                data.elems.deviceRead(),
                data.state.deviceWrite(),
                (      Deriv*)f.deviceWrite() + data.vertex0,
                (const Coord*)x.deviceRead()  + data.vertex0);
}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addDForce (Main* m, VecDeriv& df, const VecDeriv& dx, SReal kFactor, SReal /*bFactor*/)
{
    Data& data = m->data;
    df.resize(dx.size());
    Kernels::addDForce(
                data.size(),
                data.elems.deviceRead(),
                data.state.deviceRead(),
                (      Deriv*)df.deviceWrite() + data.vertex0,
                (const Deriv*)dx.deviceRead()  + data.vertex0,
                kFactor);
}


template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addKToMatrix(Main* m, sofa::linearalgebra::BaseMatrix *mat, SReal k, unsigned int &offset)
{
    Data& data = m->data;

    if (sofa::linearalgebra::CompressedRowSparseMatrix<type::Mat<3,3,double> > * crsmat = dynamic_cast<sofa::linearalgebra::CompressedRowSparseMatrix<type::Mat<3,3,double> > * >(mat))
    {
        const VecElement& elems = *m->_indexedElements;

        helper::ReadAccessor< gpu::cuda::CudaVector<GPUElementState> > state = data.state;

        // Build Matrix Block for this ForceField
        int i,j,n1, n2;
        int offd3 = offset/3;

        typename Main::Transformation Rot;
        typename Main::StiffnessMatrix JKJt,tmp;

        Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
        Rot[0][1]=Rot[0][2]=0;
        Rot[1][0]=Rot[1][2]=0;
        Rot[2][0]=Rot[2][1]=0;

        for (int ei=0; ei<data.nbElement; ++ei)
        {
            const int meshEi = data.elemReorder[ei];
            const Element& e = elems[meshEi];

            int blockIdx = ei / BSIZE;
            int threadIdx = ei % BSIZE;

            for(i=0; i<3; i++)
                for (j=0; j<3; j++)
                    Rot[j][i] = state[blockIdx].Rt[i][j][threadIdx];

            m->computeStiffnessMatrix(JKJt, tmp, m->materialsStiffnesses[meshEi], m->strainDisplacements[meshEi], Rot);
            type::Mat<3,3,double> tmpBlock[4][4];

            // find index of node 1
            for (n1=0; n1<4; n1++)
            {
                for(i=0; i<3; i++)
                {
                    for (n2=0; n2<4; n2++)
                    {
                        for (j=0; j<3; j++)
                        {
                            tmpBlock[n1][n2][i][j] = - tmp[n1*3+i][n2*3+j]*k;
                        }
                    }
                }
            }

            *crsmat->wblock(offd3 + e[0], offd3 + e[0],true) += tmpBlock[0][0];
            *crsmat->wblock(offd3 + e[0], offd3 + e[1],true) += tmpBlock[0][1];
            *crsmat->wblock(offd3 + e[0], offd3 + e[2],true) += tmpBlock[0][2];
            *crsmat->wblock(offd3 + e[0], offd3 + e[3],true) += tmpBlock[0][3];

            *crsmat->wblock(offd3 + e[1], offd3 + e[0],true) += tmpBlock[1][0];
            *crsmat->wblock(offd3 + e[1], offd3 + e[1],true) += tmpBlock[1][1];
            *crsmat->wblock(offd3 + e[1], offd3 + e[2],true) += tmpBlock[1][2];
            *crsmat->wblock(offd3 + e[1], offd3 + e[3],true) += tmpBlock[1][3];

            *crsmat->wblock(offd3 + e[2], offd3 + e[0],true) += tmpBlock[2][0];
            *crsmat->wblock(offd3 + e[2], offd3 + e[1],true) += tmpBlock[2][1];
            *crsmat->wblock(offd3 + e[2], offd3 + e[2],true) += tmpBlock[2][2];
            *crsmat->wblock(offd3 + e[2], offd3 + e[3],true) += tmpBlock[2][3];

            *crsmat->wblock(offd3 + e[3], offd3 + e[0],true) += tmpBlock[3][0];
            *crsmat->wblock(offd3 + e[3], offd3 + e[1],true) += tmpBlock[3][1];
            *crsmat->wblock(offd3 + e[3], offd3 + e[2],true) += tmpBlock[3][2];
            *crsmat->wblock(offd3 + e[3], offd3 + e[3],true) += tmpBlock[3][3];
        }
    }
    else if (sofa::linearalgebra::CompressedRowSparseMatrix<type::Mat<3,3,float> > * crsmat = dynamic_cast<sofa::linearalgebra::CompressedRowSparseMatrix<type::Mat<3,3,float> > * >(mat))
    {
        const VecElement& elems = *m->_indexedElements;

        helper::ReadAccessor< gpu::cuda::CudaVector<GPUElementState> > state = data.state;

        // Build Matrix Block for this ForceField
        int i,j,n1, n2;
        int offd3 = offset/3;

        typename Main::Transformation Rot;
        typename Main::StiffnessMatrix JKJt,tmp;

        Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
        Rot[0][1]=Rot[0][2]=0;
        Rot[1][0]=Rot[1][2]=0;
        Rot[2][0]=Rot[2][1]=0;

        for (int ei=0; ei<data.nbElement; ++ei)
        {
            const int meshEi = data.elemReorder[ei];
            const Element& e = elems[meshEi];

            int blockIdx = ei / BSIZE;
            int threadIdx = ei % BSIZE;

            for(i=0; i<3; i++)
                for (j=0; j<3; j++)
                    Rot[j][i] = state[blockIdx].Rt[i][j][threadIdx];

            m->computeStiffnessMatrix(JKJt, tmp, m->materialsStiffnesses[meshEi], m->strainDisplacements[meshEi], Rot);
            type::Mat<3,3,double> tmpBlock[4][4];

            // find index of node 1
            for (n1=0; n1<4; n1++)
            {
                for(i=0; i<3; i++)
                {
                    for (n2=0; n2<4; n2++)
                    {
                        for (j=0; j<3; j++)
                        {
                            tmpBlock[n1][n2][i][j] = - tmp[n1*3+i][n2*3+j]*k;
                        }
                    }
                }
            }

            *crsmat->wblock(offd3 + e[0], offd3 + e[0],true) += tmpBlock[0][0];
            *crsmat->wblock(offd3 + e[0], offd3 + e[1],true) += tmpBlock[0][1];
            *crsmat->wblock(offd3 + e[0], offd3 + e[2],true) += tmpBlock[0][2];
            *crsmat->wblock(offd3 + e[0], offd3 + e[3],true) += tmpBlock[0][3];

            *crsmat->wblock(offd3 + e[1], offd3 + e[0],true) += tmpBlock[1][0];
            *crsmat->wblock(offd3 + e[1], offd3 + e[1],true) += tmpBlock[1][1];
            *crsmat->wblock(offd3 + e[1], offd3 + e[2],true) += tmpBlock[1][2];
            *crsmat->wblock(offd3 + e[1], offd3 + e[3],true) += tmpBlock[1][3];

            *crsmat->wblock(offd3 + e[2], offd3 + e[0],true) += tmpBlock[2][0];
            *crsmat->wblock(offd3 + e[2], offd3 + e[1],true) += tmpBlock[2][1];
            *crsmat->wblock(offd3 + e[2], offd3 + e[2],true) += tmpBlock[2][2];
            *crsmat->wblock(offd3 + e[2], offd3 + e[3],true) += tmpBlock[2][3];

            *crsmat->wblock(offd3 + e[3], offd3 + e[0],true) += tmpBlock[3][0];
            *crsmat->wblock(offd3 + e[3], offd3 + e[1],true) += tmpBlock[3][1];
            *crsmat->wblock(offd3 + e[3], offd3 + e[2],true) += tmpBlock[3][2];
            *crsmat->wblock(offd3 + e[3], offd3 + e[3],true) += tmpBlock[3][3];
        }
    }
    else
    {
        const VecElement& elems = *m->_indexedElements;

        helper::ReadAccessor< gpu::cuda::CudaVector<GPUElementState> > state = data.state;

        // Build Matrix Block for this ForceField
        int i,j,n1, n2, row, column, ROW, COLUMN;

        typename Main::Transformation Rot;
        typename Main::StiffnessMatrix JKJt,tmp;

        Index noeud1, noeud2;

        Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
        Rot[0][1]=Rot[0][2]=0;
        Rot[1][0]=Rot[1][2]=0;
        Rot[2][0]=Rot[2][1]=0;

        for (int ei=0; ei<data.nbElement; ++ei)
        {
            const int meshEi = data.elemReorder[ei];
            const Element& e = elems[meshEi];

            int blockIdx = ei / BSIZE;
            int threadIdx = ei % BSIZE;

            for(i=0; i<3; i++)
                for (j=0; j<3; j++)
                    Rot[j][i] = state[blockIdx].Rt[i][j][threadIdx];

            m->computeStiffnessMatrix(JKJt, tmp, m->materialsStiffnesses[meshEi], m->strainDisplacements[meshEi], Rot);

            // find index of node 1
            for (n1=0; n1<4; n1++)
            {
                noeud1 = e[n1];

                for(i=0; i<3; i++)
                {
                    ROW = offset+3*noeud1+i;
                    row = 3*n1+i;
                    // find index of node 2
                    for (n2=0; n2<4; n2++)
                    {
                        noeud2 = e[n2];

                        for (j=0; j<3; j++)
                        {
                            COLUMN = offset+3*noeud2+j;
                            column = 3*n2+j;
                            mat->add(ROW, COLUMN, - tmp[row][column]*k);
                        }
                    }
                }
            }
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::getRotations(Main* m, VecReal& rotations)
{
    Data& data = m->data;
    if (data.initState.empty())
    {
        data.initState.resize((data.nbElement+BSIZE-1)/BSIZE);
        data.rotationIdx.resize(data.nbVertex);

        // Build reverse mapping: mesh element index -> GPU element index
        const auto& reorder = data.elemReorder;
        std::vector<int> meshToGpu(m->_indexedElements->size(), -1);
        for (int i = 0; i < data.nbElement; ++i)
            meshToGpu[reorder[i]] = i;

        for (int i=0; i<data.nbVertex; ++i)
        {
            data.rotationIdx[i] = meshToGpu[m->_rotationIdx[i]];
        }
        for (int i=0; i<data.nbElement; ++i)
        {
            const int ei = reorder[i];
            for (int l=0; l<3; ++l)
                for (int c=0; c<3; ++c)
                {
                    data.initState[i/BSIZE].Rt[l][c][i%BSIZE] = m->_initialRotations[ei][c][l];
                }
        }
    }
    if ((int)rotations.size() < data.nbVertex*9)
        rotations.resize(data.nbVertex*9);

    Kernels::getRotations(data.size(),
                          data.nbVertex,
                          data.initState.deviceRead(),
                          data.state.deviceRead(),
                          data.rotationIdx.deviceRead(),
                          rotations.deviceWrite());
}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::getRotations(Main* m,linearalgebra::BaseMatrix * rotations,int offset)
{
    Data& data = m->data;

    VecReal vecTmpRotation;
    vecTmpRotation.resize(data.nbVertex*9);
    data.getRotations(m, vecTmpRotation);

    for (int i=0; i<data.nbVertex; i++)
    {
        const int i9 = i*9;
        const int e = offset+i*3;
        rotations->set(e+0,e+0,vecTmpRotation[i9+0]);
        rotations->set(e+0,e+1,vecTmpRotation[i9+1]);
        rotations->set(e+0,e+2,vecTmpRotation[i9+2]);

        rotations->set(e+1,e+0,vecTmpRotation[i9+3]);
        rotations->set(e+1,e+1,vecTmpRotation[i9+4]);
        rotations->set(e+1,e+2,vecTmpRotation[i9+5]);

        rotations->set(e+2,e+0,vecTmpRotation[i9+6]);
        rotations->set(e+2,e+1,vecTmpRotation[i9+7]);
        rotations->set(e+2,e+2,vecTmpRotation[i9+8]);
    }
}

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaTetrahedronFEMForceField_ImplMethods(T) \
    template<> inline void TetrahedronFEMForceField< T >::reinit() \
{ data.reinit(this); } \
    template<> inline void TetrahedronFEMForceField< T >::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) \
{ \
    VecDeriv& f = *d_f.beginEdit(); \
    const VecCoord& x = d_x.getValue(); \
    const VecDeriv& v = d_v.getValue(); \
    data.addForce(this, f, x, v); \
    d_f.endEdit(); \
} \
    template<> inline void TetrahedronFEMForceField< T >::getRotations(VecReal & rotations) \
{ data.getRotations(this, rotations); } \
    template<> inline void TetrahedronFEMForceField< T >::getRotations(linearalgebra::BaseMatrix * rotations,int offset) \
{ data.getRotations(this, rotations,offset); } \
    template<> inline void TetrahedronFEMForceField< T >::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) \
{ \
    VecDeriv& df = *d_df.beginEdit(); \
    const VecDeriv& dx = d_dx.getValue(); \
    data.addDForce(this, df, dx, mparams->kFactor(), sofa::core::mechanicalparams::bFactor(mparams)); \
    d_df.endEdit(); \
} \
    template<> inline void TetrahedronFEMForceField< T >::addKToMatrix(sofa::linearalgebra::BaseMatrix* mat, SReal kFactor, unsigned int& offset) \
{ data.addKToMatrix(this, mat, kFactor, offset); }


CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3fTypes)

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3dTypes);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaTetrahedronFEMForceField_ImplMethods

} // namespace sofa::component::solidmechanics::fem::elastic
