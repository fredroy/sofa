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

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceField.h>

namespace sofa::gpu::cuda
{

template<class DataTypes>
class CudaKernelsHexahedronFEMForceField;

} // namespace sofa::gpu::cuda

namespace sofa::component::solidmechanics::fem::elastic
{

template <class TCoord, class TDeriv, class TReal>
class HexahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord, TDeriv, TReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef HexahedronFEMForceField<DataTypes> Main;
    typedef HexahedronFEMForceFieldInternalData<DataTypes> Data;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;


    typedef typename Main::Element Element;
    typedef typename Main::VecElement VecElement;
    typedef typename Main::Index Index;
    typedef type::Mat<6, 6, Real> MaterialStiffness;

    typedef type::Mat<24, 24, Real> ElementStiffness;

    typedef type::Mat<3, 3, Real> Transformation;

    typedef gpu::cuda::CudaKernelsHexahedronFEMForceField<DataTypes> Kernels;

    struct GPUElement
    {
        /// @name index of the 8 connected vertices
        int ia[BSIZE],ib[BSIZE],ic[BSIZE],id[BSIZE],ig[BSIZE],ih[BSIZE],ii[BSIZE],ij[BSIZE];

        /// @name initial position of the vertices in the local (rotated) coordinate system
        Real ax[BSIZE],ay[BSIZE],az[BSIZE];
        Real bx[BSIZE],by[BSIZE],bz[BSIZE];
        Real cx[BSIZE],cy[BSIZE],cz[BSIZE];
        Real dx[BSIZE],dy[BSIZE],dz[BSIZE];
        Real gx[BSIZE],gy[BSIZE],gz[BSIZE];
        Real hx[BSIZE],hy[BSIZE],hz[BSIZE];
        Real ix[BSIZE],iy[BSIZE],iz[BSIZE];
        Real jx[BSIZE],jy[BSIZE],jz[BSIZE];

    };

    gpu::cuda::CudaVector<GPUElement> elems;

     /// Symmetric element stiffness matrix: upper triangle + diagonal (36 blocks).
     /// BSIZE-interleaved: one group of KMATRIX_GROUP_SIZE Reals per BSIZE elements.
     /// Layout: [value_index * BSIZE + lane] where value_index = blockIdx*9 + mat_value
     /// (0..323), lane = elem % BSIZE.
     /// Stored as a flat CudaVector<Real> to avoid CudaVector over-reserving memory
     /// for large struct elements (SOFA_VECTOR_HOST_STEP_SIZE adds 32768 elements).
     static constexpr int KMATRIX_NBLOCKS = 36;
     static constexpr int KMATRIX_GROUP_SIZE = KMATRIX_NBLOCKS * 9 * BSIZE;

    /// Varying data associated with each element (BSIZE-interleaved rotation)
    struct GPUElementState
    {
        /// transposed rotation matrix
        Real Rt[3][3][BSIZE];
    };

    gpu::cuda::CudaVector<Real> ekmatrixData;
    gpu::cuda::CudaVector<GPUElementState> state;
    gpu::cuda::CudaVector<GPUElementState> initState;
    gpu::cuda::CudaVector<int> rotationIdx;

    int nbElement; ///< number of elements
    int vertex0; ///< index of the first vertex connected to an element
    int nbVertex; ///< number of vertices to process to compute all elements

    HexahedronFEMForceFieldInternalData() : nbElement(0), vertex0(0), nbVertex(0) {}

    void init(int nbe, int v0, int nbv)
    {
        elems.clear();
        ekmatrixData.clear();
        state.clear();
        initState.clear();
        rotationIdx.clear();

        nbElement = nbe;
        vertex0 = v0;
        nbVertex = nbv;

        const int numGroups = (nbe+BSIZE-1)/BSIZE;
        elems.resize(numGroups);
        ekmatrixData.resize(numGroups * KMATRIX_GROUP_SIZE);
        state.resize(numGroups);
    }

    int size() const { return nbElement; }

    void setE(int i, const Element& indices, type::fixed_array<Coord,8> *rotateds)
    {
       GPUElement& e = elems[i/BSIZE]; i = i % BSIZE;
       e.ia[i] = indices[0] - vertex0;
       e.ib[i] = indices[1] - vertex0;
       e.ic[i] = indices[2] - vertex0;
       e.id[i] = indices[3] - vertex0;
       e.ig[i] = indices[4] - vertex0;
       e.ih[i] = indices[5] - vertex0;
       e.ii[i] = indices[6] - vertex0;
       e.ij[i] = indices[7] - vertex0;

       e.ax[i] = (*rotateds)[0][0]; e.ay[i] = (*rotateds)[0][1]; e.az[i] = (*rotateds)[0][2];
       e.bx[i] = (*rotateds)[1][0]; e.by[i] = (*rotateds)[1][1]; e.bz[i] = (*rotateds)[1][2];
       e.cx[i] = (*rotateds)[2][0]; e.cy[i] = (*rotateds)[2][1]; e.cz[i] = (*rotateds)[2][2];
       e.dx[i] = (*rotateds)[3][0]; e.dy[i] = (*rotateds)[3][1]; e.dz[i] = (*rotateds)[3][2];
       e.gx[i] = (*rotateds)[4][0]; e.gy[i] = (*rotateds)[4][1]; e.gz[i] = (*rotateds)[4][2];
       e.hx[i] = (*rotateds)[5][0]; e.hy[i] = (*rotateds)[5][1]; e.hz[i] = (*rotateds)[5][2];
       e.ix[i] = (*rotateds)[6][0]; e.iy[i] = (*rotateds)[6][1]; e.iz[i] = (*rotateds)[6][2];
       e.jx[i] = (*rotateds)[7][0]; e.jy[i] = (*rotateds)[7][1]; e.jz[i] = (*rotateds)[7][2];

    }

    void setS(int i, const ElementStiffness& K)
    {
      const int g = i / BSIZE;
      const int li = i % BSIZE;
      Real* groupBase = &ekmatrixData[g * KMATRIX_GROUP_SIZE];
      for (int r = 0; r < 8; r++)
        for (int c = r; c < 8; c++)
        {
          type::Mat<3,3,Real> block;
          K.getsub(r*3, c*3, block);
          const int bi = 8*r - r*(r-1)/2 + (c - r);
          for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
              groupBase[(bi*9 + j*3 + k) * BSIZE + li] = block[j][k];
        }
    }

    void initPtrData(Main* /*m*/)
    {
    }

    static void reinit(Main* m);
    static void addForce(Main* m, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/);
    static void addDForce(Main* m, VecDeriv& df, const VecDeriv& dx, double kFactor/*, double bFactor*/);
};

#define CudaHexahedronFEMForceField_DeclMethods(T) \
    template<> void HexahedronFEMForceField< T >::reinit(); \
    template<> void HexahedronFEMForceField< T >::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v); \
    template<> void HexahedronFEMForceField< T >::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx); \
    template<> void HexahedronFEMForceField< T >::getRotations(linearalgebra::BaseMatrix* rotations, int offset); \

CudaHexahedronFEMForceField_DeclMethods(gpu::cuda::CudaVec3fTypes);

#undef CudaHexahedronFEMForceField_DeclMethods

} // namespace sofa::component::solidmechanics::fem::elastic
