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
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP
#include <SofaBaseLinearSolver/MatrixLinearSolver.inl>
#include <SofaBaseLinearSolver/MatrixLinearSolver[GraphScatteredMatrix].h>


namespace sofa::component::linearsolver
{

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resetSystem()
{
    if (!linearSystem->systemMatrix)
    {
        linearSystem->systemMatrix = new GraphScatteredMatrix();
    }
    if (!linearSystem->systemRHVector)
    {
        linearSystem->systemRHVector = new GraphScatteredVector(nullptr, core::VecDerivId::null());
    }
    if (!linearSystem->systemLHVector)
    {
        linearSystem->systemLHVector = new GraphScatteredVector(nullptr, core::VecDerivId::null());
    }
    linearSystem->systemRHVector->reset();
    linearSystem->systemLHVector->reset();
    linearSystem->solutionVecId = core::MultiVecDerivId::null();
    linearSystem->needInvert = true;

}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resizeSystem(Size)
{
    if (!linearSystem->systemMatrix) linearSystem->systemMatrix = new GraphScatteredMatrix();
    if (!linearSystem->systemRHVector) linearSystem->systemRHVector = new GraphScatteredVector(nullptr, core::VecDerivId::null());
    if (!linearSystem->systemLHVector) linearSystem->systemLHVector = new GraphScatteredVector(nullptr, core::VecDerivId::null());
    linearSystem->systemRHVector->reset();
    linearSystem->systemLHVector->reset();
    linearSystem->solutionVecId = core::MultiVecDerivId::null();
    linearSystem->needInvert = true;
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMatrix(GraphScatteredMatrix* matrix)
{
    linearSystem->systemMatrix = matrix;
    if (!linearSystem->systemRHVector) linearSystem->systemRHVector = new GraphScatteredVector(nullptr, core::VecDerivId::null());
    if (!linearSystem->systemLHVector) linearSystem->systemLHVector = new GraphScatteredVector(nullptr, core::VecDerivId::null());
    linearSystem->systemRHVector->reset();
    linearSystem->systemLHVector->reset();
    linearSystem->solutionVecId = core::MultiVecDerivId::null();
    linearSystem->needInvert = true;
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    resetSystem();
    linearSystem->systemMatrix->setMBKFacts(mparams);
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::rebuildSystem(double /*massFactor*/, double /*forceFactor*/)
{
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemRHVector(core::MultiVecDerivId v)
{
    linearSystem->systemRHVector->set(v);
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemLHVector(core::MultiVecDerivId v)
{
    linearSystem->solutionVecId = v;
    linearSystem->systemLHVector->set(v);

}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::solveSystem()
{
    if (linearSystem->needInvert)
    {
        this->invert(*linearSystem->systemMatrix);
        linearSystem->needInvert = false;
    }
    this->solve(*linearSystem->systemMatrix, *linearSystem->systemLHVector, *linearSystem->systemRHVector);

}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::applyConstraintForce(const sofa::core::ConstraintParams* /*cparams*/, sofa::core::MultiVecDerivId /*dx*/, const defaulttype::BaseVector* /*f*/)
{
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::computeResidual(const core::ExecParams* /*params*/,defaulttype::BaseVector* /*f*/) {
    //todo
}

template<>
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createPersistentVector()
{
    return new GraphScatteredVector(nullptr,core::VecDerivId::null());
}

template<>
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemBaseMatrix() { return nullptr; }

template<>
const core::behavior::MultiMatrixAccessor* 
MatrixLinearSolver<GraphScatteredMatrix, GraphScatteredVector, NoThreadManager>::getSystemMultiMatrixAccessor() const
{
    return nullptr;
}

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemRHBaseVector() { return nullptr; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemLHBaseVector() { return nullptr; }

// Force template instantiation
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< GraphScatteredMatrix, GraphScatteredVector, NoThreadManager >;

} // namespace sofa::component::linearsolver