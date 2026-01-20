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

#include <sofa/component/constraint/lagrangian/solver/UnbuiltGaussSeidelConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/UnbuiltConstraintProblem.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>
#include <cmath>
#include <limits>


namespace sofa::component::constraint::lagrangian::solver
{

UnbuiltGaussSeidelConstraintSolver::UnbuiltGaussSeidelConstraintSolver()
    : d_useAndersonAcceleration(initData(&d_useAndersonAcceleration, false, "useAndersonAcceleration",
        "Enable Anderson Acceleration to speed up convergence"))
    , d_andersonDepth(initData(&d_andersonDepth, 5, "andersonDepth",
        "Number of previous iterates to use for Anderson Acceleration (default: 5)"))
    , d_andersonRegularization(initData(&d_andersonRegularization, 1e-10, "andersonRegularization",
        "Regularization parameter for Anderson Acceleration least-squares solve (default: 1e-10)"))
{
}

void UnbuiltGaussSeidelConstraintSolver::solveAndersonCoefficients(
    int mk,
    int dimension,
    const sofa::type::vector<SReal>& currentResidual,
    const std::vector<sofa::type::vector<SReal>>& residualHistory,
    std::vector<SReal>& alpha) const
{
    // Anderson Acceleration: solve min ||f_k - ΔF * α||²
    // where ΔF[:,i] = f_{k-i} - f_{k-i-1} for i = 0..mk-1
    // Note: f_{k-0} = f_k = currentResidual
    //       f_{k-1}, f_{k-2}, ... are in residualHistory (most recent first)
    //
    // Using normal equations: (ΔF^T * ΔF + λI) * α = ΔF^T * f_k

    const SReal lambda = d_andersonRegularization.getValue();

    // Build ΔF matrix columns: ΔF[:,i] = f_{k-i} - f_{k-i-1}
    // i=0: f_k - f_{k-1} = currentResidual - residualHistory[0]
    // i=1: f_{k-1} - f_{k-2} = residualHistory[0] - residualHistory[1]
    // etc.
    std::vector<sofa::type::vector<SReal>> deltaF(mk);
    for (int i = 0; i < mk; ++i)
    {
        deltaF[i].resize(dimension);
        for (int j = 0; j < dimension; ++j)
        {
            SReal f_newer = (i == 0) ? currentResidual[j] : residualHistory[i-1][j];
            SReal f_older = residualHistory[i][j];
            deltaF[i][j] = f_newer - f_older;
        }
    }

    // Build AtA = ΔF^T * ΔF (mk x mk matrix) and Atb = ΔF^T * f_k (mk vector)
    std::vector<std::vector<SReal>> AtA(mk, std::vector<SReal>(mk, 0.0));
    std::vector<SReal> Atb(mk, 0.0);

    for (int i = 0; i < mk; ++i)
    {
        for (int j = 0; j < mk; ++j)
        {
            SReal dot = 0.0;
            for (int d = 0; d < dimension; ++d)
            {
                dot += deltaF[i][d] * deltaF[j][d];
            }
            AtA[i][j] = dot;
        }
        // Add regularization to diagonal
        AtA[i][i] += lambda;

        // Compute Atb[i] = deltaF[i] · currentResidual
        SReal dot = 0.0;
        for (int d = 0; d < dimension; ++d)
        {
            dot += deltaF[i][d] * currentResidual[d];
        }
        Atb[i] = dot;
    }

    // Solve AtA * alpha = Atb using Gaussian elimination with partial pivoting
    alpha.resize(mk);

    // Forward elimination
    for (int k = 0; k < mk; ++k)
    {
        // Find pivot
        int maxRow = k;
        SReal maxVal = std::abs(AtA[k][k]);
        for (int i = k + 1; i < mk; ++i)
        {
            if (std::abs(AtA[i][k]) > maxVal)
            {
                maxVal = std::abs(AtA[i][k]);
                maxRow = i;
            }
        }

        // Swap rows
        if (maxRow != k)
        {
            std::swap(AtA[k], AtA[maxRow]);
            std::swap(Atb[k], Atb[maxRow]);
        }

        // Check for singular matrix
        if (std::abs(AtA[k][k]) < 1e-14)
        {
            // Matrix is singular, return zero coefficients
            std::fill(alpha.begin(), alpha.end(), 0.0);
            return;
        }

        // Eliminate column
        for (int i = k + 1; i < mk; ++i)
        {
            SReal factor = AtA[i][k] / AtA[k][k];
            for (int j = k + 1; j < mk; ++j)
            {
                AtA[i][j] -= factor * AtA[k][j];
            }
            Atb[i] -= factor * Atb[k];
        }
    }

    // Back substitution
    for (int k = mk - 1; k >= 0; --k)
    {
        SReal sum = Atb[k];
        for (int j = k + 1; j < mk; ++j)
        {
            sum -= AtA[k][j] * alpha[j];
        }
        alpha[k] = sum / AtA[k][k];
    }
}

void UnbuiltGaussSeidelConstraintSolver::doSolve(GenericConstraintProblem * problem , SReal timeout)
{
    UnbuiltConstraintProblem* c_current_cp = dynamic_cast<UnbuiltConstraintProblem*>(problem);
    if (c_current_cp == nullptr)
    {
        msg_error()<<"Constraint problem must derive from UnbuiltConstraintProblem";
        return;
    }

    SCOPED_TIMER_VARNAME(unbuiltGaussSeidelTimer, "ConstraintsUnbuiltGaussSeidel");


    if(!c_current_cp->getDimension())
    {
        c_current_cp->currentError = 0.0;
        c_current_cp->currentIterations = 0;
        return;
    }

    SReal t0 = (SReal)sofa::helper::system::thread::CTime::getTime();
    SReal timeScale = 1.0 / (SReal)sofa::helper::system::thread::CTime::getTicksPerSec();

    SReal *dfree = c_current_cp->getDfree();
    SReal *force = c_current_cp->getF();
    SReal **w = c_current_cp->getW();
    SReal tol = c_current_cp->tolerance;

    SReal *d = c_current_cp->_d.ptr();

    unsigned int iter = 0, nb = 0;

    SReal error=0.0;

    bool convergence = false;
    sofa::type::vector<SReal> tempForces;
    if(c_current_cp->sor != 1.0) tempForces.resize(c_current_cp->getDimension());

    if(c_current_cp->scaleTolerance && !c_current_cp->allVerified)
        tol *= c_current_cp->getDimension();


    for(int i=0; i<c_current_cp->getDimension(); )
    {
        if(!c_current_cp->constraintsResolutions[i])
        {
            msg_warning() << "Bad size of constraintsResolutions in GenericConstraintSolver" ;
            c_current_cp->setDimension(i);
            break;
        }
        c_current_cp->constraintsResolutions[i]->init(i, w, force);
        i += c_current_cp->constraintsResolutions[i]->getNbLines();
    }
    // Note: force array is now initialized by GenericConstraintSolver::computeInitialGuess()
    // for hot-start support. Do not zero forces here.

    bool showGraphs = false;
    sofa::type::vector<SReal>* graph_residuals = nullptr;
    std::map < std::string, sofa::type::vector<SReal> > *graph_forces = nullptr, *graph_violations = nullptr;
    sofa::type::vector<SReal> tabErrors;


    showGraphs = d_computeGraphs.getValue();

    if(showGraphs)
    {
        graph_forces = d_graphForces.beginEdit();
        graph_forces->clear();

        graph_violations = d_graphViolations.beginEdit();
        graph_violations->clear();

        graph_residuals = &(*d_graphErrors.beginEdit())["Error"];
        graph_residuals->clear();
    }

    tabErrors.resize(c_current_cp->getDimension());

    // temporary buffers
    std::vector<SReal> errF;
    std::vector<SReal> tempF;

    // Anderson Acceleration setup
    const bool useAA = d_useAndersonAcceleration.getValue();
    const int andersonDepth = d_andersonDepth.getValue();
    const int dimension = c_current_cp->getDimension();

    // Storage for current iterate before GS sweep (x_k) and current g(x_k)
    sofa::type::vector<SReal> x_k;
    sofa::type::vector<SReal> g_k;
    // Storage for current residual f_k = g_k - x_k
    sofa::type::vector<SReal> currentResidual;
    // Track previous error for restart detection
    SReal prevError = std::numeric_limits<SReal>::max();

    if (useAA)
    {
        x_k.resize(dimension);
        g_k.resize(dimension);
        currentResidual.resize(dimension);

        // Reset history if dimension changed
        if (dimension != m_lastDimension)
        {
            m_lastDimension = dimension;
            m_residualHistory.clear();
            m_gHistory.clear();
        }
    }

    for(iter=0; iter < static_cast<unsigned int>(c_current_cp->maxIterations); iter++)
    {
        bool constraintsAreVerified = true;
        if(c_current_cp->sor != 1.0)
        {
            std::copy_n(force, dimension, tempForces.begin());
        }

        // Anderson: Save current force as x_k before GS sweep
        if (useAA)
        {
            std::copy_n(force, dimension, x_k.begin());
        }

        error=0.0;
        for (auto it_c = c_current_cp->constraints_sequence.begin(); it_c != c_current_cp->constraints_sequence.end(); )  // increment of it_c realized at the end of the loop
        {
            const auto j = *it_c;
            //1. nbLines provide the dimension of the constraint
            nb = c_current_cp->constraintsResolutions[j]->getNbLines();

            //2. for each line we compute the actual value of d
            //   (a)d is set to dfree
            if(nb > errF.size())
            {
                errF.resize(nb);
            }
            std::copy_n(&force[j], nb, errF.begin());
            std::copy_n(&dfree[j], nb, &d[j]);

            //   (b) contribution of forces are added to d
            for (auto* el : c_current_cp->cclist_elems[j])
            {
                if (el)
                    el->addConstraintDisplacement(d, j, j+nb-1);
            }

            //3. the specific resolution of the constraint(s) is called
            c_current_cp->constraintsResolutions[j]->resolution(j, w, d, force, dfree);

            //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
            SReal contraintError = 0.0;
            if(nb > 1)
            {
                for(unsigned int l=0; l<nb; l++)
                {
                    SReal lineError = 0.0;
                    for (unsigned int m=0; m<nb; m++)
                    {
                        SReal dofError = w[j+l][j+m] * (force[j+m] - errF[m]);
                        lineError += dofError * dofError;
                    }
                    lineError = sqrt(lineError);
                    if(lineError > tol)
                        constraintsAreVerified = false;

                    contraintError += lineError;
                }
            }
            else
            {
                contraintError = fabs(w[j][j] * (force[j] - errF[0]));
                if(contraintError > tol)
                    constraintsAreVerified = false;
            }

            if(c_current_cp->constraintsResolutions[j]->getTolerance())
            {
                if(contraintError > c_current_cp->constraintsResolutions[j]->getTolerance())
                    constraintsAreVerified = false;
                contraintError *= tol / c_current_cp->constraintsResolutions[j]->getTolerance();
            }

            error += contraintError;
            tabErrors[j] = contraintError;

            //5. the force is updated for the constraint corrections
            bool update = false;
            for(unsigned int l=0; l<nb; l++)
                update |= (force[j+l] || errF[l]);

            if(update)
            {
                if (nb > tempF.size())
                {
                    tempF.resize(nb);
                }
                std::copy_n(&force[j], nb, tempF.begin());
                for(unsigned int l=0; l<nb; l++)
                {
                    force[j+l] -= errF[l]; // DForce
                }

                for (auto* el : c_current_cp->cclist_elems[j])
                {
                    if (el)
                        el->setConstraintDForce(force, j, j+nb-1, update);
                }

                std::copy_n(tempF.begin(), nb, &force[j]);
            }
            std::advance(it_c, nb);
        }

        if(showGraphs)
        {
            for(int j=0; j<c_current_cp->getDimension(); j++)
            {
                std::ostringstream oss;
                oss << "f" << j;

                sofa::type::vector<SReal>& graph_force = (*graph_forces)[oss.str()];
                graph_force.push_back(force[j]);

                sofa::type::vector<SReal>& graph_violation = (*graph_violations)[oss.str()];
                graph_violation.push_back(d[j]);
            }

            graph_residuals->push_back(error);
        }

        // Anderson Acceleration: apply extrapolation based on history
        if (useAA)
        {
            // After GS sweep, force contains g(x_k)
            // Save g_k before any modification
            std::copy_n(force, dimension, g_k.begin());

            // Compute current residual f_k = g(x_k) - x_k
            for (int j = 0; j < dimension; ++j)
            {
                currentResidual[j] = g_k[j] - x_k[j];
            }

            // Check for restart: if error increased significantly, clear history
            if (error > prevError * 1.5 && !m_residualHistory.empty())
            {
                m_residualHistory.clear();
                m_gHistory.clear();
            }
            prevError = error;

            // Apply Anderson Acceleration if we have enough history
            // Need at least 1 historical residual to compute 1 difference
            const int historySize = static_cast<int>(m_residualHistory.size());
            const int mk = std::min(historySize, andersonDepth);
            if (mk > 0)
            {
                std::vector<SReal> alpha;
                solveAndersonCoefficients(mk, dimension, currentResidual, m_residualHistory, alpha);

                // Compute the sum of alpha coefficients - should be bounded for stability
                SReal alphaSum = 0.0;
                for (int i = 0; i < mk; ++i)
                {
                    alphaSum += std::abs(alpha[i]);
                }

                // Only apply if coefficients are reasonable (not too large)
                // Large coefficients indicate the least-squares problem is ill-conditioned
                if (alphaSum < 2.0)
                {
                    // Compute the AA correction for each constraint
                    // and update both force AND the constraint corrections
                    sofa::type::vector<SReal> aaCorrection(dimension, 0.0);

                    // Compute total correction: -sum_{i=0}^{mk-1} alpha_i * (g_{k-i} - g_{k-i-1})
                    for (int i = 0; i < mk; ++i)
                    {
                        for (int j = 0; j < dimension; ++j)
                        {
                            SReal g_newer = (i == 0) ? g_k[j] : m_gHistory[i-1][j];
                            SReal g_older = m_gHistory[i][j];
                            aaCorrection[j] -= alpha[i] * (g_newer - g_older);
                        }
                    }

                    // Apply correction to force
                    for (int j = 0; j < dimension; ++j)
                    {
                        force[j] += aaCorrection[j];
                    }

                    // CRITICAL: Also update constraint corrections to match
                    // This ensures consistency between force and the internal state
                    for (auto it_c = c_current_cp->constraints_sequence.begin();
                         it_c != c_current_cp->constraints_sequence.end(); )
                    {
                        const auto j = *it_c;
                        const unsigned int nb = c_current_cp->constraintsResolutions[j]->getNbLines();

                        // Check if any correction is non-zero for this constraint
                        bool hasCorrection = false;
                        for (unsigned int l = 0; l < nb; ++l)
                        {
                            if (aaCorrection[j + l] != 0.0)
                            {
                                hasCorrection = true;
                                break;
                            }
                        }

                        if (hasCorrection)
                        {
                            // Temporarily set force to the AA delta
                            sofa::type::vector<SReal> savedForce(nb);
                            for (unsigned int l = 0; l < nb; ++l)
                            {
                                savedForce[l] = force[j + l];
                                force[j + l] = aaCorrection[j + l];
                            }

                            // Update constraint corrections with the AA delta
                            for (auto* el : c_current_cp->cclist_elems[j])
                            {
                                if (el)
                                    el->setConstraintDForce(force, j, j + nb - 1, true);
                            }

                            // Restore force to AA result
                            for (unsigned int l = 0; l < nb; ++l)
                            {
                                force[j + l] = savedForce[l];
                            }
                        }

                        std::advance(it_c, nb);
                    }
                }
            }

            // Store current residual and g_k in history (insert at front, most recent first)
            m_residualHistory.insert(m_residualHistory.begin(), currentResidual);
            m_gHistory.insert(m_gHistory.begin(), g_k);

            // Trim history to max depth + 1 (need mk+1 entries for mk differences)
            if (static_cast<int>(m_residualHistory.size()) > andersonDepth + 1)
            {
                m_residualHistory.pop_back();
                m_gHistory.pop_back();
            }
        }

        if(c_current_cp->sor != 1.0)
        {
            for(int j=0; j<dimension; j++)
                force[j] = c_current_cp->sor * force[j] + (1-c_current_cp->sor) * tempForces[j];
        }
        if(timeout)
        {
            SReal t1 = (SReal)sofa::helper::system::thread::CTime::getTime();
            SReal dt = (t1 - t0)*timeScale;

            if(dt > timeout)
            {
                c_current_cp->currentError = error;
                c_current_cp->currentIterations = iter+1;
                return;
            }
        }
        else if(c_current_cp->allVerified)
        {
            if(constraintsAreVerified)
            {
                convergence = true;
                break;
            }
        }
        else if(error < tol)
        {
            convergence = true;
            break;
        }
    }



    sofa::helper::AdvancedTimer::valSet("GS iterations", c_current_cp->currentIterations);

    c_current_cp->result_output(this, force, error, iter, convergence);

    if(showGraphs)
    {
        d_graphErrors.endEdit();

        sofa::type::vector<SReal>& graph_constraints = (*d_graphConstraints.beginEdit())["Constraints"];
        graph_constraints.clear();

        for(int j=0; j<c_current_cp->getDimension(); )
        {
            nb = c_current_cp->constraintsResolutions[j]->getNbLines();

            if(tabErrors[j])
                graph_constraints.push_back(tabErrors[j]);
            else if(c_current_cp->constraintsResolutions[j]->getTolerance())
                graph_constraints.push_back(c_current_cp->constraintsResolutions[j]->getTolerance());
            else
                graph_constraints.push_back(tol);

            j += nb;
        }
        d_graphConstraints.endEdit();

        d_graphForces.endEdit();
    }
}

void registerUnbuiltGaussSeidelConstraintSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components using an Unbuilt version of the Gauss-Seidel iterative method")
        .add< UnbuiltGaussSeidelConstraintSolver >());
}

}
