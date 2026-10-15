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

#include <gtest/gtest.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/BoyceAndArruda.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/Costa.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/NeoHookean.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/Ogden.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/StableNeoHookean.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/STVenantKirchhoff.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/VerondaWestman.h>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>
#include <sofa/testing/BaseTest.h>


namespace sofa
{

using namespace component::solidmechanics::fem::hyperelastic::material;

StrainInformation<defaulttype::Vec3Types>::MatrixSym
generatePositiveDefiniteMatrix(sofa::testing::LinearCongruentialRandomGenerator& lcg)
{
    using Real = defaulttype::Vec3Types::Coord::value_type;

    Eigen::Matrix<Real, 3, 3> A;
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            A(i, j) = lcg.generateInRange(-10., 10.);
        }
    }

    A = 0.5 * (A + A.transpose()); // Ensuring symmetry

    // Making the matrix positive definite
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 3, 3> > eigensolver(A);
    const Eigen::Matrix<Real, 3, 1>& eigenvalues = eigensolver.eigenvalues();
    const Real min_eigval = eigenvalues.minCoeff();
    if (min_eigval <= 0)
    {
        A += Eigen::Matrix<Real, 3, 3>::Identity() * (-min_eigval + 1e-6); // Adding a small value to ensure positive definiteness
    }

    StrainInformation<defaulttype::Vec3Types>::MatrixSym W;
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i <= j; i++)
        {
            W(i, j) = (A(i, j) + A(j, i)) / 2.0;
        }
    }
    return W;
}

/**
 * Computes the invariants of the right Cauchy-Green deformation tensor.
 * The sign of J = det(F) cannot be recovered from C: it is given by Jsign.
 */
void computeStrainInformation(StrainInformation<defaulttype::Vec3Types>& strain,
                              defaulttype::Vec3Types::Coord::value_type Jsign = 1)
{
    strain.trC = sofa::type::trace(strain.deformationTensor);

    //det(C)=J^2
    strain.J = std::copysign(std::sqrt(sofa::type::determinant(strain.deformationTensor)), Jsign);
}

/// Strain information computed from a deformation gradient F, with a signed J = det(F)
StrainInformation<defaulttype::Vec3Types> computeStrainInformation(
    const type::Mat<3, 3, defaulttype::Vec3Types::Coord::value_type>& F)
{
    StrainInformation<defaulttype::Vec3Types> strain;
    StrainInformation<defaulttype::Vec3Types>::MatrixSym::Mat2Sym(F.transposed() * F, strain.deformationTensor);
    strain.trC = sofa::type::trace(strain.deformationTensor);
    strain.J = sofa::type::determinant(F);
    return strain;
}

/// Largest absolute value of the entries of a tensor, but at least 1
template<class Tensor>
SReal tensorScale(const Tensor& t)
{
    SReal scale = 1;
    for (sofa::Size i = 0; i < Tensor::size(); ++i)
    {
        scale = std::max(scale, static_cast<SReal>(std::abs(t[i])));
    }
    return scale;
}

/// Lamé parameters {mu, lambda} from the Young modulus and the Poisson ratio
std::vector<SReal> lameParameters(SReal youngModulus, SReal poissonRatio)
{
    const SReal mu = youngModulus / (2 * (1 + poissonRatio));
    const SReal lambda = youngModulus * poissonRatio / ((1 + poissonRatio) * (1 - 2 * poissonRatio));
    return {mu, lambda};
}

SReal perturbedStrainEnergy(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters,
    const StrainInformation<defaulttype::Vec3Types>& strain_0,
    defaulttype::Vec3Types::Coord::value_type h,
    const sofa::Size i)
{
    StrainInformation<defaulttype::Vec3Types> strain = strain_0;

    //off-diagonal terms: division by 2 because of the symmetric nature of the
    //tensor data structure
    if (i != 0 && i != 2 && i != 5)
    {
        h /= 2;
    }

    strain.deformationTensor[i] += h;
    computeStrainInformation(strain, strain_0.J);

    return material.getStrainEnergy(&strain, materialParameters);
}

StrainInformation<defaulttype::Vec3Types>::MatrixSym perturbedPK2(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters,
    const StrainInformation<defaulttype::Vec3Types>& strain_0,
    defaulttype::Vec3Types::Coord::value_type h,
    const sofa::Size i)
{
    StrainInformation<defaulttype::Vec3Types> strain = strain_0;

    //off-diagonal terms: division by 2 because of the symmetric nature of the
    //tensor data structure
    if (i != 0 && i != 2 && i != 5)
    {
        h /= 2;
    }

    strain.deformationTensor[i] += h;
    computeStrainInformation(strain, strain_0.J);

    StrainInformation<defaulttype::Vec3Types>::MatrixSym PK2;
    material.deriveSPKTensor(&strain, materialParameters, PK2);
    return PK2;
}

void testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters,
    StrainInformation<defaulttype::Vec3Types> strain)
{
    using Real = defaulttype::Vec3Types::Coord::value_type;

    StrainInformation<defaulttype::Vec3Types>::MatrixSym PK2;
    material.deriveSPKTensor(&strain, materialParameters, PK2);

    //the round-off error of the finite differences is proportional to the magnitude of the values
    const Real tolerance = 1e-7 * tensorScale(PK2);

    static constexpr Real h = 1e-6;

    for (sofa::Size i = 0; i < StrainInformation<defaulttype::Vec3Types>::MatrixSym::size(); ++i)
    // for (sofa::Size i : {0, 2, 5}) //only the diagonal terms
    {
        const Real psiPlus =
            perturbedStrainEnergy(material, materialParameters, strain, h, i);

        const Real psiMinus =
            perturbedStrainEnergy(material, materialParameters, strain, -h, i);

        //approximation of dPsi/dC
        const Real centralDifference = (psiPlus - psiMinus) / (2 * h);

        //PK2 = 2 * dPsi/dC
        const Real PK2Approx = 2 * centralDifference;

        //compare the approximation of PK2 with PK2
        EXPECT_NEAR(PK2[i], PK2Approx, tolerance) << "i = " << i;
    }
}

StrainInformation<defaulttype::Vec3Types> generateStrainInformation()
{
    StrainInformation<defaulttype::Vec3Types> strain;

    //random right Cauchy-Green deformation tensor
    sofa::testing::LinearCongruentialRandomGenerator lcg(96547);
    strain.deformationTensor = generatePositiveDefiniteMatrix(lcg);

    computeStrainInformation(strain);
    return strain;
}

void testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters)
{
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters, generateStrainInformation());
}

void testElasticityTensorFromSecondPiolaKirchhoff(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters,
    StrainInformation<defaulttype::Vec3Types> strain)
{
    using Real = defaulttype::Vec3Types::Coord::value_type;
    using MatrixSym = StrainInformation<defaulttype::Vec3Types>::MatrixSym;
    using Matrix6 = HyperelasticMaterial<defaulttype::Vec3Types>::Matrix6;

    Matrix6 elasticityTensor;
    material.ElasticityTensor(&strain, materialParameters, elasticityTensor);

    SReal scale = 1;
    for (sofa::Size i = 0; i < Matrix6::size(); ++i)
    {
        scale = std::max(scale, tensorScale(elasticityTensor[i]));
    }
    const Real tolerance = 1e-7 * scale;

    static constexpr Real h = 1e-6;

    for (sofa::Size i = 0; i < Matrix6::size(); ++i)
    {
        const MatrixSym pk2Plus =
                perturbedPK2(material, materialParameters, strain, h, i);

        const MatrixSym pk2Minus =
                perturbedPK2(material, materialParameters, strain, -h, i);

        //approximation of dPK2/dC
        const MatrixSym centralDifference = (pk2Plus - pk2Minus) / (2 * h);

        //ElasticityTensor = 2 * dPK2/dC
        MatrixSym elasticityTensorApprox = 2 * centralDifference;

        //compare the approximation of the elasticity tensor
        for (sofa::Size j = 0; j < Matrix6::size(); ++j)
        {
            // Off-diagonal terms are stored doubled; the tensor approximation must be scaled accordingly
            if (j == 1 || j == 3 || j == 4) elasticityTensorApprox[j] *= 2.;
            EXPECT_NEAR(elasticityTensor(i,j), elasticityTensorApprox[j], tolerance) << "i = " << i << ", j = " << j;
        }
    }
}

void testElasticityTensorFromSecondPiolaKirchhoff(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters)
{
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters, generateStrainInformation());
}

void testApplyElasticityTensor(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters,
    StrainInformation<defaulttype::Vec3Types> strain,
    const StrainInformation<defaulttype::Vec3Types>::MatrixSym& e)
{
    using MatrixSym = StrainInformation<defaulttype::Vec3Types>::MatrixSym;
    using Matrix6 = HyperelasticMaterial<defaulttype::Vec3Types>::Matrix6;

    sofa::type::Vec<6, SReal> eVec;
    for (int i = 0; i < 6; ++i)
    {
        eVec[i] = e[i];
    }

    //first method to obtain the application of the elasticity tensor: matrix-vector product
    Matrix6 elasticityTensor;
    material.ElasticityTensor(&strain, materialParameters, elasticityTensor);

    const sofa::type::Vec<6, SReal> appliedTensor_1 = elasticityTensor * eVec;

    //second method to obtain the application of the elasticity tensor: matrix-free product
    MatrixSym appliedTensor_2;
    material.applyElasticityTensor(&strain, materialParameters, e, appliedTensor_2);

    const SReal tolerance = 1e-3 * tensorScale(appliedTensor_1);
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_NEAR(appliedTensor_1[i], 2 * appliedTensor_2[i], tolerance) << "i = " << i;
    }
}

void testApplyElasticityTensor(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters)
{
    StrainInformation<defaulttype::Vec3Types> strain;

    //random right Cauchy-Green deformation tensor
    sofa::testing::LinearCongruentialRandomGenerator lcg(96547);
    strain.deformationTensor = generatePositiveDefiniteMatrix(lcg);

    //another random-generated symmetric second-order tensor
    const StrainInformation<defaulttype::Vec3Types>::MatrixSym e =
            generatePositiveDefiniteMatrix(lcg);

    computeStrainInformation(strain);

    testApplyElasticityTensor(material, materialParameters, strain, e);
}

TEST(HyperelasticMaterial, PK2_StVenantKirchhoff)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    STVenantKirchhoff<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_NeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    NeoHookean<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_Ogden)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 2., 3., 2.};

    Ogden<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_ArrudaBoyce)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    BoyceAndArruda<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_VerondaWestman)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1., 1.};

    VerondaWestman<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_Costa)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1., 1., 1., 1., 1., 1., 1.};

    Costa<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_StableNeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    StableNeoHookean<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}



TEST(HyperelasticMaterial, ElasticityTensor_StVenantKirchhoff)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    STVenantKirchhoff<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_NeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    NeoHookean<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_Ogden)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 2., 3., 2.};

    Ogden<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_StableNeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    StableNeoHookean<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_ArrudaBoyce)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    BoyceAndArruda<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_VerondaWestman)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1., 1.};

    VerondaWestman<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}





TEST(HyperelasticMaterial, applyElasticityTensor_StVenantKirchhoff)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    STVenantKirchhoff<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_NeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    NeoHookean<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_Ogden)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 2., 3., 2.};

    Ogden<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_StableNeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    StableNeoHookean<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_ArrudaBoyce)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    BoyceAndArruda<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_VerondaWestman)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1., 1.};

    VerondaWestman<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}


/////////////////////////////////////////////////////////////////////////////
// Tests based on the deformation gradient F, with a signed J = det(F)
/////////////////////////////////////////////////////////////////////////////

using Matrix3 = type::Mat<3, 3, defaulttype::Vec3Types::Coord::value_type>;

/// First Piola-Kirchhoff stress tensor P = F * S
Matrix3 firstPiolaKirchhoff(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters,
    const Matrix3& F)
{
    StrainInformation<defaulttype::Vec3Types> strain = computeStrainInformation(F);
    StrainInformation<defaulttype::Vec3Types>::MatrixSym PK2;
    material.deriveSPKTensor(&strain, materialParameters, PK2);
    return F * PK2;
}

/// Directional derivative of P = F * S along dF, using applyElasticityTensor:
/// dP = dF * S + F * dS, with dS = dS/dC : (dF^T F + F^T dF)
Matrix3 firstPiolaKirchhoffDifferential(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters,
    const Matrix3& F, const Matrix3& dF)
{
    using MatrixSym = StrainInformation<defaulttype::Vec3Types>::MatrixSym;

    StrainInformation<defaulttype::Vec3Types> strain = computeStrainInformation(F);
    MatrixSym PK2;
    material.deriveSPKTensor(&strain, materialParameters, PK2);

    MatrixSym dC;
    MatrixSym::Mat2Sym(dF.transposed() * F + F.transposed() * dF, dC);

    MatrixSym dPK2;
    material.applyElasticityTensor(&strain, materialParameters, dC, dPK2);

    return dF * PK2 + F * dPK2;
}

Matrix3 generateMatrix(sofa::testing::LinearCongruentialRandomGenerator& lcg, SReal amplitude)
{
    Matrix3 M;
    for (sofa::Size i = 0; i < 3; ++i)
    {
        for (sofa::Size j = 0; j < 3; ++j)
        {
            M(i, j) = lcg.generateInRange(-amplitude, amplitude);
        }
    }
    return M;
}

/// Deformation gradients: a stretched one (J > 0), and inverted ones (J < 0)
std::vector<Matrix3> generateDeformationGradients()
{
    sofa::testing::LinearCongruentialRandomGenerator lcg(4567);

    Matrix3 stretched = Matrix3::Identity() + generateMatrix(lcg, 0.3);

    // reflection of the first row: det(F) changes sign
    Matrix3 inverted = Matrix3::Identity() + generateMatrix(lcg, 0.3);
    inverted[0] *= -1;

    // strongly compressed through the origin
    Matrix3 crushed = Matrix3::Identity() + generateMatrix(lcg, 0.1);
    crushed[2] *= -0.2;

    return {stretched, inverted, crushed};
}

std::vector<std::vector<SReal>> stableNeoHookeanParameterSets()
{
    return {
        {1., 1.},
        lameParameters(1., 0.3),
        lameParameters(1e5, 0.49) // mu = 33557.05, lambda = 1644295.3
    };
}

TEST(HyperelasticMaterial, StableNeoHookean_restStateIsStressFree)
{
    StableNeoHookean<defaulttype::Vec3Types> material{};

    for (const auto& parameters : stableNeoHookeanParameterSets())
    {
        MaterialParameters<defaulttype::Vec3Types> materialParameters;
        materialParameters.parameterArray = parameters;
        const SReal scale = parameters[0] + parameters[1];

        StrainInformation<defaulttype::Vec3Types> strain = computeStrainInformation(Matrix3::Identity());

        EXPECT_NEAR(material.getStrainEnergy(&strain, materialParameters), 0, 1e-12 * scale);

        StrainInformation<defaulttype::Vec3Types>::MatrixSym PK2;
        material.deriveSPKTensor(&strain, materialParameters, PK2);
        for (sofa::Size i = 0; i < PK2.size(); ++i)
        {
            EXPECT_NEAR(PK2[i], 0, 1e-12 * scale) << "i = " << i;
        }
    }
}

TEST(HyperelasticMaterial, StableNeoHookean_linearizationAtRestIsLinearElasticity)
{
    StableNeoHookean<defaulttype::Vec3Types> material{};
    sofa::testing::LinearCongruentialRandomGenerator lcg(8910);

    for (const auto& parameters : stableNeoHookeanParameterSets())
    {
        MaterialParameters<defaulttype::Vec3Types> materialParameters;
        materialParameters.parameterArray = parameters;
        const SReal mu = parameters[0];
        const SReal lambda = parameters[1];

        // displacement gradient
        const Matrix3 G = generateMatrix(lcg, 1.);

        // linear elasticity: sigma = 2 mu eps + lambda tr(eps) I, with eps = (G + G^T) / 2
        const Matrix3 linearStress = mu * (G + G.transposed()) + lambda * sofa::type::trace(G) * Matrix3::Identity();

        // finite differences of P around F = I
        static constexpr SReal h = 1e-6;
        const Matrix3 PPlus = firstPiolaKirchhoff(material, materialParameters, Matrix3::Identity() + h * G);
        const Matrix3 PMinus = firstPiolaKirchhoff(material, materialParameters, Matrix3::Identity() - h * G);
        const Matrix3 dPApprox = (PPlus - PMinus) / (2 * h);

        // analytic differential of P at F = I
        const Matrix3 dP = firstPiolaKirchhoffDifferential(material, materialParameters, Matrix3::Identity(), G);

        const SReal tolerance = 1e-6 * (mu + lambda);
        for (sofa::Size i = 0; i < 3; ++i)
        {
            for (sofa::Size j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(dPApprox(i, j), linearStress(i, j), tolerance) << "i = " << i << ", j = " << j << ", mu = " << mu << ", lambda = " << lambda;
                EXPECT_NEAR(dP(i, j), linearStress(i, j), tolerance) << "i = " << i << ", j = " << j << ", mu = " << mu << ", lambda = " << lambda;
            }
        }
    }
}

TEST(HyperelasticMaterial, StableNeoHookean_closedForm)
{
    StableNeoHookean<defaulttype::Vec3Types> material{};

    for (const auto& parameters : stableNeoHookeanParameterSets())
    {
        MaterialParameters<defaulttype::Vec3Types> materialParameters;
        materialParameters.parameterArray = parameters;
        const SReal mu = parameters[0];
        const SReal lambda = parameters[1] + mu;
        const SReal alpha = 1 + mu / lambda;

        for (const Matrix3& F : generateDeformationGradients())
        {
            // signed volume change
            const SReal J = sofa::type::determinant(F);

            // W = mu/2 (I_C - 3) + lambda'/2 (J - alpha)^2 - mu^2 / (2 lambda'), lambda' = lambda + mu
            StrainInformation<defaulttype::Vec3Types> strain = computeStrainInformation(F);
            const SReal expectedEnergy = mu / 2 * (strain.trC - 3) + lambda / 2 * (J - alpha) * (J - alpha) - mu * mu / (2 * lambda);
            EXPECT_NEAR(material.getStrainEnergy(&strain, materialParameters), expectedEnergy, 1e-12 * std::abs(expectedEnergy))
                << "J = " << J << ", mu = " << parameters[0] << ", lambda = " << parameters[1];

            // P = mu F + lambda' (J - alpha) cof(F), cof(F) = J F^-T
            const Matrix3 expectedP = mu * F + (lambda * (J - alpha) * J) * F.inverted().transposed();
            const Matrix3 P = firstPiolaKirchhoff(material, materialParameters, F);

            SReal scale = 1;
            for (sofa::Size i = 0; i < 3; ++i)
            {
                scale = std::max(scale, tensorScale(expectedP[i]));
            }

            for (sofa::Size i = 0; i < 3; ++i)
            {
                for (sofa::Size j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(P(i, j), expectedP(i, j), 1e-10 * scale)
                        << "i = " << i << ", j = " << j << ", J = " << J << ", mu = " << parameters[0] << ", lambda = " << parameters[1];
                }
            }
        }
    }
}

TEST(HyperelasticMaterial, StableNeoHookean_firstPiolaKirchhoffFromStrainEnergy)
{
    StableNeoHookean<defaulttype::Vec3Types> material{};

    for (const auto& parameters : stableNeoHookeanParameterSets())
    {
        MaterialParameters<defaulttype::Vec3Types> materialParameters;
        materialParameters.parameterArray = parameters;

        for (const Matrix3& F : generateDeformationGradients())
        {
            const Matrix3 P = firstPiolaKirchhoff(material, materialParameters, F);

            SReal scale = 1;
            for (sofa::Size i = 0; i < 3; ++i)
            {
                scale = std::max(scale, tensorScale(P[i]));
            }

            static constexpr SReal h = 1e-6;
            for (sofa::Size i = 0; i < 3; ++i)
            {
                for (sofa::Size j = 0; j < 3; ++j)
                {
                    Matrix3 FPlus = F, FMinus = F;
                    FPlus(i, j) += h;
                    FMinus(i, j) -= h;

                    StrainInformation<defaulttype::Vec3Types> strainPlus = computeStrainInformation(FPlus);
                    StrainInformation<defaulttype::Vec3Types> strainMinus = computeStrainInformation(FMinus);

                    // P = dW/dF
                    const SReal PApprox = (material.getStrainEnergy(&strainPlus, materialParameters)
                        - material.getStrainEnergy(&strainMinus, materialParameters)) / (2 * h);

                    EXPECT_NEAR(P(i, j), PApprox, 1e-7 * scale)
                        << "i = " << i << ", j = " << j << ", J = " << sofa::type::determinant(F) << ", mu = " << parameters[0] << ", lambda = " << parameters[1];
                }
            }
        }
    }
}

TEST(HyperelasticMaterial, StableNeoHookean_tangentFromFirstPiolaKirchhoff)
{
    StableNeoHookean<defaulttype::Vec3Types> material{};
    sofa::testing::LinearCongruentialRandomGenerator lcg(1112);

    for (const auto& parameters : stableNeoHookeanParameterSets())
    {
        MaterialParameters<defaulttype::Vec3Types> materialParameters;
        materialParameters.parameterArray = parameters;

        for (const Matrix3& F : generateDeformationGradients())
        {
            const Matrix3 dF = generateMatrix(lcg, 1.);

            const Matrix3 dP = firstPiolaKirchhoffDifferential(material, materialParameters, F, dF);

            static constexpr SReal h = 1e-6;
            const Matrix3 dPApprox = (firstPiolaKirchhoff(material, materialParameters, F + h * dF)
                - firstPiolaKirchhoff(material, materialParameters, F - h * dF)) / (2 * h);

            SReal scale = 1;
            for (sofa::Size i = 0; i < 3; ++i)
            {
                scale = std::max(scale, tensorScale(dP[i]));
            }

            for (sofa::Size i = 0; i < 3; ++i)
            {
                for (sofa::Size j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(dP(i, j), dPApprox(i, j), 1e-7 * scale)
                        << "i = " << i << ", j = " << j << ", J = " << sofa::type::determinant(F) << ", mu = " << parameters[0] << ", lambda = " << parameters[1];
                }
            }
        }
    }
}

TEST(HyperelasticMaterial, StableNeoHookean_tensorsInvertedAndIncompressible)
{
    StableNeoHookean<defaulttype::Vec3Types> material{};
    sofa::testing::LinearCongruentialRandomGenerator lcg(1314);

    for (const auto& parameters : stableNeoHookeanParameterSets())
    {
        MaterialParameters<defaulttype::Vec3Types> materialParameters;
        materialParameters.parameterArray = parameters;

        for (const Matrix3& F : generateDeformationGradients())
        {
            const StrainInformation<defaulttype::Vec3Types> strain = computeStrainInformation(F);

            // the finite differences on C keep the sign of J
            testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters, strain);
            testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters, strain);
            testApplyElasticityTensor(material, materialParameters, strain, generatePositiveDefiniteMatrix(lcg));
        }
    }
}

// BaseTest installs the message handler required by EXPECT_MSG_EMIT
using HyperelasticMaterialMessages = sofa::testing::BaseTest;

TEST_F(HyperelasticMaterialMessages, StableNeoHookean_flatElement)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 2., 3.};

    StableNeoHookean<defaulttype::Vec3Types> material{};

    Matrix3 F = Matrix3::Identity();
    F(2, 2) = 0;
    StrainInformation<defaulttype::Vec3Types> strain = computeStrainInformation(F);

    StrainInformation<defaulttype::Vec3Types>::MatrixSym PK2;
    {
        EXPECT_MSG_EMIT(Warning);
        material.deriveSPKTensor(&strain, materialParameters, PK2);
    }

    // C^-1 cannot be computed: only the deviatoric term remains, S = mu * I
    StrainInformation<defaulttype::Vec3Types>::MatrixSym expected;
    expected.identity();
    expected = materialParameters.parameterArray[0] * expected;
    for (sofa::Size i = 0; i < PK2.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(PK2[i], expected[i]) << "i = " << i;
    }

    // the warning is emitted only once
    {
        EXPECT_MSG_NOEMIT(Warning);
        material.deriveSPKTensor(&strain, materialParameters, PK2);
    }
}

TEST(HyperelasticMaterial, StableNeoHookean_float)
{
    using MatrixSym = StableNeoHookean<defaulttype::Vec3fTypes>::MatrixSym;

    MaterialParameters<defaulttype::Vec3fTypes> materialParameters;
    materialParameters.parameterArray = { 1.f, 1.f};

    StableNeoHookean<defaulttype::Vec3fTypes> material{};

    StrainInformation<defaulttype::Vec3fTypes> strain;
    strain.deformationTensor.identity();
    strain.trC = 3;
    strain.J = 1;

    EXPECT_NEAR(material.getStrainEnergy(&strain, materialParameters), 0.f, 1e-6f);

    MatrixSym PK2;
    material.deriveSPKTensor(&strain, materialParameters, PK2);

    MatrixSym input;
    input.identity();
    MatrixSym output;
    material.applyElasticityTensor(&strain, materialParameters, input, output);

    StableNeoHookean<defaulttype::Vec3fTypes>::Matrix6 elasticityTensor;
    material.ElasticityTensor(&strain, materialParameters, elasticityTensor);

    for (sofa::Size i = 0; i < MatrixSym::size(); ++i)
    {
        EXPECT_NEAR(PK2[i], 0.f, 1e-6f) << "i = " << i;
    }

    // at rest, dS/dC : I = mu * I + lambda / 2 * tr(I) * I
    EXPECT_FLOAT_EQ(output(0, 0), 2.5f);
    EXPECT_FLOAT_EQ(output(0, 1), 0.f);
}

}
