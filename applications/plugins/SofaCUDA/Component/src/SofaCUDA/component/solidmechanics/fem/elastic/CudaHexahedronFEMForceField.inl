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
#include <SofaCUDA/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceField.inl>
#include <sofa/gpu/cuda/mycuda.h>

namespace sofa::gpu::cuda
{

extern "C"
{

void HexahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, const void* elems, void* state, const void* kmatrix, void* f, const void* x);
void HexahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, const void* elems, const void* state, const void* kmatrix, void* df, const void* dx, double kFactor);
void HexahedronFEMForceFieldCuda3f_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
void HexahedronFEMForceFieldCuda3f_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);

} // extern "C"

template<>
class CudaKernelsHexahedronFEMForceField<CudaVec3fTypes>
{
public:
    static void addForce(unsigned int nbElem, const void* elems, void* state, const void* kmatrix, void* f, const void* x)
    {   HexahedronFEMForceFieldCuda3f_addForce(nbElem, elems, state, kmatrix, f, x); }
    static void addDForce(unsigned int nbElem, const void* elems, const void* state, const void* kmatrix, void* df, const void* dx, double kFactor)
    {   HexahedronFEMForceFieldCuda3f_addDForce(nbElem, elems, state, kmatrix, df, dx, kFactor); }
    static void getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
    {   HexahedronFEMForceFieldCuda3f_getRotations(nbElem, nbVertex, initState, state, rotationIdx, rotations); }
    static void getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations)
    {   HexahedronFEMForceFieldCuda3f_getElementRotations(nbElem, rotationsAos, rotations); }
};

} // namespace sofa::gpu::cuda


namespace sofa::component::solidmechanics::fem::elastic
{

using namespace gpu::cuda;
using namespace core::behavior;

template<class TCoord, class TDeriv, class TReal>
void HexahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::reinit(Main* m)
{
  Data& data = *m->data;
  m->setMethod(m->LARGE);

  const VecCoord& p = m->mstate->read(sofa::core::vec_id::read_access::restPosition)->getValue();
  m->d_initialPoints.setValue(p);

  m->_materialsStiffnesses.resize( m->getIndexedElements()->size() );
  m->d_elementStiffnesses.beginEdit()->resize( m->getIndexedElements()->size() );

  m->_rotatedInitialElements.resize(m->getIndexedElements()->size());
  m->_rotations.resize( m->getIndexedElements()->size() );
  m->_initialrotations.resize( m->getIndexedElements()->size() );

    unsigned int i = 0;
    typename VecElement::const_iterator it;
    for(it = m->getIndexedElements()->begin() ; it != m->getIndexedElements()->end() ; ++it, ++i)
    {
        m->computeMaterialStiffness(i);
        m->initLarge(i, *it);
    }

  const VecElement& elems = *m->getIndexedElements();

  std::vector<int> activeElems;
  for (unsigned int i=0;i<elems.size();i++)
  {
        activeElems.push_back(i);
  }

  std::map<int,int> nelems;
  for (unsigned int i=0;i<activeElems.size();i++)
  {
      int ei = activeElems[i];
      const Element& e = elems[ei];
      for (unsigned int j=0;j<e.size();j++)
          ++nelems[e[j]];
  }
  int v0 = 0;
  int nbv = 0;
  if (!nelems.empty())
  {
      v0 = nelems.begin()->first;
      nbv = nelems.rbegin()->first - v0 + 1;
  }
  data.init(activeElems.size(), v0, nbv);

  for (unsigned int i=0;i<activeElems.size();i++)
  {
      int ei = activeElems[i];
      const Element& e = elems[ei];

      data.setE(ei, e, &(m->_rotatedInitialElements[ei]));
      data.setS(ei, m->d_elementStiffnesses.getValue()[i]);
  }

  m->d_elementStiffnesses.endEdit();

}

template<class TCoord, class TDeriv, class TReal>
void HexahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addForce(Main* m, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/)
{
  Data& data = *m->data;

    f.resize(x.size());
    Kernels::addForce(
        data.size(),
        data.elems.deviceRead(),
        data.state.deviceWrite(),
        data.ekmatrixData.deviceRead(),
        (      Deriv*)f.deviceWrite() + data.vertex0,
        (const Coord*)x.deviceRead()  + data.vertex0);
} // addForce

template<class TCoord, class TDeriv, class TReal>
void HexahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addDForce (Main* m, VecDeriv& df, const VecDeriv& dx, double kFactor/*, double bFactor*/)
{
    Data& data = *m->data;
    df.resize(dx.size());

    Kernels::addDForce(
        data.size(),
        data.elems.deviceRead(),
        data.state.deviceRead(),
        data.ekmatrixData.deviceRead(),
        (      Deriv*)df.deviceWrite() + data.vertex0,
        (const Deriv*)dx.deviceRead()  + data.vertex0,
         kFactor);

} // addDForce


template<>
void HexahedronFEMForceField< gpu::cuda::CudaVec3fTypes >::reinit()	{ data->reinit(this);}
template<>
void HexahedronFEMForceField< gpu::cuda::CudaVec3fTypes >::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    data->addForce(this, f, x, v);

    d_f.endEdit();
}
template<>
void HexahedronFEMForceField< gpu::cuda::CudaVec3fTypes >::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    const double kFactor = mparams->kFactor();

    data->addDForce(this, df, dx, kFactor/*, bFactor*/);

    d_df.endEdit();
}

template<>
const HexahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::Transformation& HexahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::getElementRotation(const unsigned elemidx) {
    // Read from BSIZE-interleaved state buffer on the CPU side
    const int block = elemidx / BSIZE;
    const int thread = elemidx % BSIZE;
    const auto* stateHost = data->state.hostRead();
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            _rotations[elemidx][i][j] = stateHost[block].Rt[i][j][thread];
    return _rotations[elemidx];
}

template<>
void HexahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::getRotations(linearalgebra::BaseMatrix * rotations,int offset)
{
    typedef HexahedronFEMForceFieldInternalData<gpu::cuda::CudaVec3fTypes> Data;
    typedef Data::Kernels Kernels;
    Data& d = *data;

    // Lazy-init initState and rotationIdx on first call
    if (d.initState.empty())
    {
        d.initState.resize((d.nbElement+BSIZE-1)/BSIZE);
        d.rotationIdx.resize(d.nbVertex);

        // Build rotationIdx: for each vertex, store one element index
        const VecElement& elems = *getIndexedElements();
        for (unsigned int ei=0; ei<elems.size(); ei++)
        {
            const Element& e = elems[ei];
            for (unsigned int j=0; j<e.size(); j++)
            {
                int v = e[j] - d.vertex0;
                if (v >= 0 && v < d.nbVertex)
                    d.rotationIdx[v] = ei;
            }
        }

        // Fill initState from _initialrotations (BSIZE-interleaved)
        for (int ei=0; ei<d.nbElement; ei++)
        {
            for (int r=0; r<3; r++)
                for (int c=0; c<3; c++)
                    d.initState[ei/BSIZE].Rt[r][c][ei%BSIZE] = _initialrotations[ei][r][c];
        }
    }

    // Compute per-vertex rotations on GPU
    const auto nbdof = this->mstate->getSize();
    gpu::cuda::CudaVector<float> gpuRotations;
    gpuRotations.resize(d.nbVertex * 9);

    Kernels::getRotations(d.size(),
                          d.nbVertex,
                          d.initState.deviceRead(),
                          d.state.deviceRead(),
                          d.rotationIdx.deviceRead(),
                          gpuRotations.deviceWrite());

    // Read results back to CPU and fill the BaseMatrix
    helper::ReadAccessor< gpu::cuda::CudaVector<float> > rotData = gpuRotations;

    if (auto* diag = dynamic_cast<linearalgebra::RotationMatrix<float> *>(rotations))
    {
        for (unsigned int i=0; i<nbdof; ++i)
        {
            const int vi = i - d.vertex0;
            if (vi >= 0 && vi < d.nbVertex)
            {
                const int i9 = vi * 9;
                for(int k=0; k<9; k++)
                    diag->getVector()[i*9 + k] = rotData[i9 + k];
            }
            else
            {
                // Identity for vertices outside element range
                for(int k=0; k<9; k++)
                    diag->getVector()[i*9 + k] = (k % 4 == 0) ? 1.0f : 0.0f;
            }
        }
    }
    else if (auto* diag = dynamic_cast<linearalgebra::RotationMatrix<double> *>(rotations))
    {
        for (unsigned int i=0; i<nbdof; ++i)
        {
            const int vi = i - d.vertex0;
            if (vi >= 0 && vi < d.nbVertex)
            {
                const int i9 = vi * 9;
                for(int k=0; k<9; k++)
                    diag->getVector()[i*9 + k] = rotData[i9 + k];
            }
            else
            {
                for(int k=0; k<9; k++)
                    diag->getVector()[i*9 + k] = (k % 4 == 0) ? 1.0 : 0.0;
            }
        }
    }
    else
    {
        for (unsigned int i=0; i<nbdof; ++i)
        {
            const int vi = i - d.vertex0;
            const int e = offset+i*3;
            if (vi >= 0 && vi < d.nbVertex)
            {
                const int i9 = vi * 9;
                rotations->set(e+0,e+0,rotData[i9+0]); rotations->set(e+0,e+1,rotData[i9+1]); rotations->set(e+0,e+2,rotData[i9+2]);
                rotations->set(e+1,e+0,rotData[i9+3]); rotations->set(e+1,e+1,rotData[i9+4]); rotations->set(e+1,e+2,rotData[i9+5]);
                rotations->set(e+2,e+0,rotData[i9+6]); rotations->set(e+2,e+1,rotData[i9+7]); rotations->set(e+2,e+2,rotData[i9+8]);
            }
            else
            {
                rotations->set(e+0,e+0,1); rotations->set(e+0,e+1,0); rotations->set(e+0,e+2,0);
                rotations->set(e+1,e+0,0); rotations->set(e+1,e+1,1); rotations->set(e+1,e+2,0);
                rotations->set(e+2,e+0,0); rotations->set(e+2,e+1,0); rotations->set(e+2,e+2,1);
            }
        }
    }
}

} // namespace sofa::component::solidmechanics::fem::elastic
