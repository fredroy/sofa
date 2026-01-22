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
#include <limits>


namespace sofa::component::constraint::lagrangian::solver
{


void UnbuiltGaussSeidelConstraintSolver::doSolve(GenericConstraintProblem * problem , SReal timeout)
{
    UnbuiltConstraintProblem* c_current_cp = dynamic_cast<UnbuiltConstraintProblem*>(problem);
    if (c_current_cp == nullptr)
    {
        msg_error()<<"Constraint problem must derive from UnbuiltConstraintProblem";
        return;
    }

    SCOPED_TIMER_VARNAME(unbuiltGaussSeidelTimer, "ConstraintsUnbuiltGaussSeidel");

    // Cache dimension to avoid repeated virtual calls
    const int dimension = c_current_cp->getDimension();

    if(!dimension)
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

    // Use pre-allocated buffers from problem
    std::vector<SReal>& tempForces = c_current_cp->tempForces;
    std::vector<SReal>& errF = c_current_cp->errF;
    std::vector<SReal>& dForce = c_current_cp->dForce;
    std::vector<SReal>& tabErrors = c_current_cp->tabErrors;

    if(c_current_cp->sor != 1.0) tempForces.resize(dimension);

    if(c_current_cp->scaleTolerance && !c_current_cp->allVerified)
        tol *= dimension;

    // Pre-compute squared tolerance for deferred sqrt optimization
    const SReal tolSquared = tol * tol;

    // Count constraint blocks and initialize stability tracking
    int numConstraintBlocks = 0;
    for(int i=0; i<dimension; )
    {
        if(!c_current_cp->constraintsResolutions[i])
        {
            msg_warning() << "Bad size of constraintsResolutions in GenericConstraintSolver" ;
            c_current_cp->setDimension(i);
            break;
        }
        c_current_cp->constraintsResolutions[i]->init(i, w, force);
        numConstraintBlocks++;
        i += c_current_cp->constraintsResolutions[i]->getNbLines();
    }
    // Note: force array is now initialized by GenericConstraintSolver::computeInitialGuess()
    // for hot-start support. Do not zero forces here.

    // Initialize per-constraint stability tracking
    c_current_cp->stableIterCount.resize(dimension);
    std::fill(c_current_cp->stableIterCount.begin(), c_current_cp->stableIterCount.end(), 0);

    bool showGraphs = false;
    sofa::type::vector<SReal>* graph_residuals = nullptr;
    std::map < std::string, sofa::type::vector<SReal> > *graph_forces = nullptr, *graph_violations = nullptr;

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

    tabErrors.resize(dimension);

    // Stagnation detection: stop early if error stops improving
    SReal prevError = std::numeric_limits<SReal>::max();
    unsigned int stagnationCount = 0;
    constexpr unsigned int STAGNATION_THRESHOLD = 10;  // Stop after 10 iterations without significant improvement
    constexpr SReal IMPROVEMENT_THRESHOLD = 0.99;      // Consider stagnant if error >= 99% of previous

    for(iter=0; iter < static_cast<unsigned int>(c_current_cp->maxIterations); iter++)
    {
        bool constraintsAreVerified = true;
        if(c_current_cp->sor != 1.0)
        {
            std::copy_n(force, dimension, tempForces.begin());
        }

        error=0.0;

        // Iterate using index for better cache performance with vector
        const auto& constraints_seq = c_current_cp->constraints_sequence;
        auto& stableCount = c_current_cp->stableIterCount;

        for (size_t seq_idx = 0; seq_idx < constraints_seq.size(); )
        {
            const unsigned int j = constraints_seq[seq_idx];
            //1. nbLines provide the dimension of the constraint
            nb = c_current_cp->constraintsResolutions[j]->getNbLines();

            // Skip stable constraints that have converged (no force change for several iterations)
            // Only skip if: after first iteration, stable for threshold iterations, AND error is below tolerance
            const SReal constraintTol = c_current_cp->constraintsResolutions[j]->getTolerance()
                                        ? c_current_cp->constraintsResolutions[j]->getTolerance()
                                        : tol;

            // Aggressively skip constraints with very low error (essentially converged)
            // Skip if error is below 1% of tolerance, regardless of stability
            if (iter > 0 && tabErrors[j] < constraintTol * 0.01)
            {
                error += tabErrors[j];
                seq_idx += nb;
                continue;
            }

            if (iter > 0 && stableCount[j] >= UnbuiltConstraintProblem::STABLE_SKIP_THRESHOLD
                && tabErrors[j] < constraintTol)
            {
                // Still accumulate the last known error
                error += tabErrors[j];
                seq_idx += nb;
                continue;
            }

            //2. for each line we compute the actual value of d
            //   (a)d is set to dfree
            if(nb > errF.size())
            {
                errF.resize(nb);
                dForce.resize(nb);
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
            //   Compute dForce = force - errF (will be reused in step 5)
            for(unsigned int l=0; l<nb; l++)
            {
                dForce[l] = force[j+l] - errF[l];
            }

            SReal contraintError = 0.0;
            if(nb > 1)
            {
                // Defer sqrt: accumulate squared errors, compare with squared tolerance
                for(unsigned int l=0; l<nb; l++)
                {
                    SReal lineErrorSquared = 0.0;
                    for (unsigned int m=0; m<nb; m++)
                    {
                        SReal dofError = w[j+l][j+m] * dForce[m];
                        lineErrorSquared += dofError * dofError;
                    }
                    if(lineErrorSquared > tolSquared)
                        constraintsAreVerified = false;

                    contraintError += sqrt(lineErrorSquared);
                }
            }
            else
            {
                contraintError = fabs(w[j][j] * dForce[0]);
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
            //   setConstraintDForce expects df[begin..end], so we must use the full force array
            //   Use pre-computed dForce to avoid recomputing delta
            bool update = false;
            for(unsigned int l=0; l<nb; l++)
                update |= (dForce[l] != 0.0);

            if(update)
            {
                // Temporarily put dForce into force array (setConstraintDForce expects df[j..j+nb-1])
                for(unsigned int l=0; l<nb; l++)
                    force[j+l] = dForce[l];

                for (auto* el : c_current_cp->cclist_elems[j])
                {
                    if (el)
                        el->setConstraintDForce(force, j, j+nb-1, update);
                }

                // Restore force: new_force = errF + dForce
                for(unsigned int l=0; l<nb; l++)
                    force[j+l] = errF[l] + dForce[l];

                // Reset stability counter when force changes
                stableCount[j] = 0;
            }
            else
            {
                // Increment stability counter when force doesn't change
                stableCount[j]++;
            }
            seq_idx += nb;
        }

        if(showGraphs)
        {
            for(int j=0; j<dimension; j++)
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

        if(c_current_cp->sor != 1.0)
        {
            for(int j=0; j<dimension; j++)
                force[j] = c_current_cp->sor * force[j] + (1-c_current_cp->sor) * tempForces[j];
        }

        // Stagnation detection: check if error is improving
        if (error >= prevError * IMPROVEMENT_THRESHOLD)
        {
            stagnationCount++;
            if (stagnationCount >= STAGNATION_THRESHOLD)
            {
                // Error stopped improving, exit early
                break;
            }
        }
        else
        {
            stagnationCount = 0;  // Reset on improvement
        }
        prevError = error;

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

        for(int j=0; j<dimension; )
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
