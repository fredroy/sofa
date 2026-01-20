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

#include <sofa/component/constraint/lagrangian/solver/UnbuiltConstraintSolver.h>
#include <sofa/core/behavior/ConstraintResolution.h>

namespace sofa::component::constraint::lagrangian::solver
{
class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API UnbuiltGaussSeidelConstraintSolver : public UnbuiltConstraintSolver
{
public:
    SOFA_CLASS(UnbuiltGaussSeidelConstraintSolver, UnbuiltConstraintSolver);

    // Anderson Acceleration parameters
    Data<bool> d_useAndersonAcceleration; ///< Enable Anderson Acceleration to speed up convergence
    Data<int> d_andersonDepth; ///< Number of previous iterates to use for Anderson Acceleration (default: 5)
    Data<SReal> d_andersonRegularization; ///< Regularization parameter for Anderson Acceleration least-squares solve (default: 1e-10)

    UnbuiltGaussSeidelConstraintSolver();

    virtual void doSolve(GenericConstraintProblem * problem , SReal timeout = 0.0) override;

protected:
    // Anderson Acceleration history storage (simple vectors, most recent first)
    std::vector<sofa::type::vector<SReal>> m_residualHistory;  ///< Previous residuals f_i = g_i - x_i
    std::vector<sofa::type::vector<SReal>> m_gHistory;         ///< Previous g(x) values
    int m_lastDimension{0}; ///< Last problem dimension (to detect changes)

    /// Solve least-squares problem for Anderson coefficients
    void solveAndersonCoefficients(
        int mk,  // number of history entries to use
        int dimension,
        const sofa::type::vector<SReal>& currentResidual,
        const std::vector<sofa::type::vector<SReal>>& residualHistory,
        std::vector<SReal>& alpha) const;
};
}