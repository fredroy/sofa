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
#include <sofa/component/solidmechanics/fem/elastic/config.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::fem::elastic
{
template<class DataTypes>
class TriangleFEMUtils
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord;
    typedef typename DataTypes::Deriv    Deriv;
    typedef typename Coord::value_type   Real;

    typedef type::Vec<6, Real> Displacement;					    ///< the displacement vector
    typedef type::Mat<3, 3, Real> MaterialStiffness;				    ///< the matrix of material stiffness
    typedef type::Mat<3, 3, Real > Transformation;						///< matrix for rigid transformations like rotations
    typedef type::Mat<6, 3, Real> StrainDisplacement;				    ///< the strain-displacement matrix

    typedef sofa::core::topology::BaseMeshTopology::Index Index;

    ////////////// small displacements method
    static void computeDisplacementSmall(Displacement& D, const type::fixed_array<Coord, 3>& rotatedInitCoord, const Coord& pAB, const Coord& pAC)
    {
        D[0] = 0;
        D[1] = 0;
        D[2] = rotatedInitCoord[1][0] - pAB[0];
        D[3] = rotatedInitCoord[1][1] - pAB[1];
        D[4] = rotatedInitCoord[2][0] - pAC[0];
        D[5] = rotatedInitCoord[2][1] - pAC[1];
    }

    ////////////// large displacements method
    static void computeDisplacementLarge(Displacement& D, const Transformation& R_0_2, const type::fixed_array<Coord, 3>& rotatedInitCoord,const Coord& pA, const Coord& pB, const Coord& pC)
    {
        // positions of the deformed and displaced triangle in its local frame
        const Coord deforme_b = R_0_2 * (pB - pA);
        const Coord deforme_c = R_0_2 * (pC - pA);

        // displacements in the local frame
        D[0] = 0;
        D[1] = 0;
        D[2] = rotatedInitCoord[1][0] - deforme_b[0];
        D[3] = 0;
        D[4] = rotatedInitCoord[2][0] - deforme_c[0];
        D[5] = rotatedInitCoord[2][1] - deforme_c[1];
    }
    static void computeRotationLarge(Transformation& r, const Coord& pA, const Coord& pB, const Coord& pC)
    {
        const Coord edgex = (pB - pA).normalized();
        Coord edgey = pC - pA;
        const Coord edgez = cross(edgex, edgey).normalized();
        edgey = cross(edgez, edgex); //edgey is unit vector because edgez and edgex are orthogonal unit vectors

        r[0][0] = edgex[0];
        r[0][1] = edgex[1];
        r[0][2] = edgex[2];
        r[1][0] = edgey[0];
        r[1][1] = edgey[1];
        r[1][2] = edgey[2];
        r[2][0] = edgez[0];
        r[2][1] = edgez[1];
        r[2][2] = edgez[2];
    }

    static void computeForceLarge(Displacement& F, const StrainDisplacement& J, const type::Vec<3, Real>& stress)
    {
        F[0] = J[0][0] * stress[0] + J[0][2] * stress[2];
        F[1] = J[1][1] * stress[1] + J[1][2] * stress[2];
        F[2] = J[2][0] * stress[0] + J[2][2] * stress[2];
        F[3] = J[3][1] * stress[1] + J[3][2] * stress[2];
        F[4] = J[4][2] * stress[2];
        F[5] = J[5][1] * stress[1];
    }
    
    // in global coordinate
    static void computeStrainDisplacementGlobal(StrainDisplacement& J, const Coord& pA, const Coord& pB, const Coord& pC)
    {
        const Coord ab_cross_ac = cross(pB - pA, pC - pA);
        const Real determinant = ab_cross_ac.norm();

        if (fabs(determinant) < std::numeric_limits<Real>::epsilon())
        {
            msg_error("TriangleFEMUtils") << "Null determinant in computeStrainDisplacementGlobal: " << determinant;
            throw std::logic_error("Division by zero exception in computeStrainDisplacementGlobal ");
        }

        const Real invDet = 1 / determinant;

        const Real x13 = (pA[0] - pC[0]) * invDet;
        const Real x21 = (pB[0] - pA[0]) * invDet;
        const Real x32 = (pC[0] - pB[0]) * invDet;
        const Real y12 = (pA[1] - pB[1]) * invDet;
        const Real y23 = (pB[1] - pC[1]) * invDet;
        const Real y31 = (pC[1] - pA[1]) * invDet;

        J[0][0] = y23;
        J[0][1] = 0;
        J[0][2] = x32;

        J[1][0] = 0;
        J[1][1] = x32;
        J[1][2] = y23;

        J[2][0] = y31;
        J[2][1] = 0;
        J[2][2] = x13;

        J[3][0] = 0;
        J[3][1] = x13;
        J[3][2] = y31;

        J[4][0] = y12;
        J[4][1] = 0;
        J[4][2] = x21;

        J[5][0] = 0;
        J[5][1] = x21;
        J[5][2] = y12;
    }
    // in local coordinate, a = Coord (0, 0, 0)
    static void computeStrainDisplacementLocal(StrainDisplacement& J, const Coord& pB, const Coord& pC)
    {
        // local computation taking into account that a = [0, 0, 0], b = [x, 0, 0], c = [y, y, 0]
        const Real determinant = pB[0] * pC[1];

        if (fabs(determinant) < std::numeric_limits<Real>::epsilon())
        {
            msg_error("TriangleFEMUtils") << "Null determinant in computeStrainDisplacementGlobal: " << determinant;
            throw std::logic_error("Division by zero exception in computeStrainDisplacementLocal");
        }
        const Real invDet = 1 / determinant;

        J[0][0] = J[1][2] = -pC[1] * invDet;
        J[0][2] = J[1][1] = (pC[0] - pB[0]) * invDet;
        J[2][0] = J[3][2] = pC[1] * invDet;
        J[2][2] = J[3][1] = -pC[0] * invDet;
        J[4][0] = J[5][2] = 0;
        J[4][2] = J[5][1] = pB[0] * invDet;
        J[1][0] = J[3][0] = J[5][0] = J[0][1] = J[2][1] = J[4][1] = 0;
    }

    // Compute strain, if full is set to true, full matrix multiplication is performed not taking into account potential 0 values
    static void computeStrain(type::Vec<3, Real>& strain, const StrainDisplacement& J, const Displacement& D, bool fullMethod = false)
    {
        if (fullMethod) // _anisotropicMaterial or SMALL case
        {
            strain = J.multTranspose(D);
        }
        else
        {
            // Use directly J to avoid computing Jt
            strain[0] = J[0][0] * D[0] + J[2][0] * D[2];
            strain[1] = J[1][1] * D[1] + J[3][1] * D[3] + J[5][1] * D[5];
            strain[2] = J[0][2] * D[0] + J[1][2] * D[1] + J[2][2] * D[2] + J[3][2] * D[3] + J[4][2] * D[4];
        }
    }
    // Compute stress, if full is set to true, full matrix multiplication is performed not taking into account potential 0 values
    static void computeStress(type::Vec<3, Real>& stress, const MaterialStiffness& K, const type::Vec<3, Real>& strain, bool fullMethod = false)
    {
        if (fullMethod) // _anisotropicMaterial or SMALL case
        {
            stress = K * strain;
        }
        else
        {
            stress[0] = K[0][0] * strain[0] + K[0][1] * strain[1];
            stress[1] = K[1][0] * strain[0] + K[1][1] * strain[1];
            stress[2] = K[2][2] * strain[2];
        }
    }
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGLEFEMUTILS_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TriangleFEMUtils<defaulttype::Vec3Types>;

#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGLEFEMUTILS_CPP)


} //namespace sofa::component::solidmechanics::fem::elastic
