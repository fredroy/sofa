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

#include <fstream>
#include <sofa/component/linearsystem/config.h>
#include <sofa/component/linearsystem/MappingGraph.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/VecId.h>

#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <Eigen/Sparse>

namespace sofa::component::linearsystem
{

template<class BlockType>
Eigen::Map<Eigen::SparseMatrix<BlockType, Eigen::RowMajor> > makeEigenMap(const linearalgebra::CompressedRowSparseMatrix<BlockType>& matrix)
{
    using EigenMap = Eigen::Map<Eigen::SparseMatrix<BlockType, Eigen::RowMajor> >;
    return Eigen::Map<Eigen::SparseMatrix<BlockType, Eigen::RowMajor> >(
            static_cast<typename EigenMap::Index>(matrix.rows()),
            static_cast<typename EigenMap::Index>(matrix.cols()),
            static_cast<typename EigenMap::Index>(matrix.getColsValue().size()),
            (typename EigenMap::StorageIndex*)matrix.rowBegin.data(),
            (typename EigenMap::StorageIndex*)matrix.colsIndex.data(),
            (typename EigenMap::Scalar*)matrix.colsValue.data());
}

/**
 * Add the local matrix which has been built locally to the main global matrix, using the Eigen library
 *
 * @remark Eigen manages the matrix operations better than CompressedRowSparseMatrix. In terms of performances, it is
 * preferable to go with Eigen.
 *
 * @param mstatePair The mapped mechanical state which the local matrix is associated
 * @param mappedMatrix The local matrix
 * @param jacobians The required mapping jacobians to project from a mechanical state toward the top most mechanical states
 * @param mappingGraph The mapping graph used to know the relationships between mechanical states. In particular, it
 * is used to know where in the global matrix the local matrix must be added.
 * @param globalMatrix Matrix in which the local matrix is added.
 */
template<class BlockType>
void addMappedMatrixToGlobalMatrixEigen(
    sofa::type::fixed_array<core::behavior::BaseMechanicalState*, 2> mstatePair,
    linearalgebra::CompressedRowSparseMatrix<BlockType>* mappedMatrix,
    sofa::type::fixed_array<
            std::map< core::behavior::BaseMechanicalState*, std::shared_ptr<linearalgebra::CompressedRowSparseMatrix<BlockType> > >,
            2> jacobians,
    const MappingGraph& mappingGraph,
    linearalgebra::BaseMatrix* globalMatrix)
{
    if (!mappedMatrix)
    {
        return;
    }

    if (mappedMatrix->rows() == 0 || mappedMatrix->cols() == 0)
    {
        return;
    }

    if (mappedMatrix->getColsValue().empty() && mappedMatrix->btemp.empty())
    {
        return;
    }

    mappedMatrix->fullRows();

    const auto KMap = makeEigenMap(*mappedMatrix);

    // nb rows of K = size of first mechanical state
    msg_error_when(mappedMatrix->rows() != mstatePair[0]->getMatrixSize(), "MatrixMapping")
        << "[K] Incompatible matrix size [rows] " << mappedMatrix->rows() << " " << mstatePair[0]->getMatrixSize();
    // nb cols of K = size of second mechanical state
    msg_error_when(mappedMatrix->cols() != mstatePair[1]->getMatrixSize(), "MatrixMapping")
        << "[K] Incompatible matrix size [cols] " << mappedMatrix->cols() << " " << mstatePair[1]->getMatrixSize();

    const auto inputs1 = mappingGraph.getTopMostMechanicalStates(mstatePair[0]);
    const auto inputs2 = mappingGraph.getTopMostMechanicalStates(mstatePair[1]);

    std::set<core::behavior::BaseMechanicalState*> inputs;
    inputs.insert(inputs1.begin(), inputs1.end());
    inputs.insert(inputs2.begin(), inputs2.end());

    std::set< std::pair<core::behavior::BaseMechanicalState*, core::behavior::BaseMechanicalState*> > uniquePairs;
    std::map< core::behavior::BaseMechanicalState*, Eigen::SparseMatrix<BlockType, Eigen::RowMajor> > inputJacobiansStiffness; // stores J^T * K
    for (auto* a : inputs)
    {
        for (auto* b : inputs)
        {
            uniquePairs.insert({a, b});
        }
    }

    for (const auto& [a, b] : uniquePairs)
    {
        const auto it1 = jacobians[0].find(a);
        const bool hasJ1 = it1 != jacobians[0].end();
        if (hasJ1)
        {
            // nb rows of J1 = size of first mechanical state
            msg_error_when(it1->second->rows() != mstatePair[0]->getMatrixSize(), "MatrixMapping")
                << "[J1] Incompatible matrix size [rows] " << it1->second->rows() << " " << mstatePair[0]->getMatrixSize();
            msg_error_when(it1->second->cols() != a->getMatrixSize(), "MatrixMapping")
                << "[J1] Incompatible matrix size [cols] " << it1->second->cols() << " " << a->getMatrixSize();
        }

        const auto it2 = jacobians[1].find(b);
        const bool hasJ2 = it2 != jacobians[1].end();
        if (hasJ2)
        {
            // nb rows of J2 = size of second mechanical state
            msg_error_when(it2->second->rows() != mstatePair[1]->getMatrixSize(), "MatrixMapping")
                << "[J2] Incompatible matrix size [rows] " << it2->second->rows() << " " << mstatePair[1]->getMatrixSize();
            msg_error_when(it2->second->cols() != b->getMatrixSize(), "MatrixMapping")
                << "[J2] Incompatible matrix size [cols] " << it2->second->cols() << " " << b->getMatrixSize();
        }

        Eigen::SparseMatrix<BlockType, Eigen::RowMajor> JT_K_J;

        if (hasJ1 && hasJ2)
        {
            const auto JMap1 = makeEigenMap(*it1->second);
            const auto JMap2 = makeEigenMap(*it2->second);
            JT_K_J = JMap1.transpose() * KMap * JMap2;
        }
        else if (hasJ1 && !hasJ2)
        {
            const auto JMap1 = makeEigenMap(*it1->second);
            JT_K_J = JMap1.transpose() * KMap;
        }
        else if (!hasJ1 && hasJ2)
        {
            const auto JMap2 = makeEigenMap(*it2->second);
            JT_K_J = KMap * JMap2;
        }
        else
        {
            JT_K_J = KMap;
        }

        const auto positionInGlobalMatrix = mappingGraph.getPositionInGlobalMatrix(a, b);

        for (int k = 0; k < JT_K_J.outerSize(); ++k)
        {
            for (typename Eigen::SparseMatrix<BlockType, Eigen::RowMajor>::InnerIterator it(JT_K_J,k); it; ++it)
            {
                globalMatrix->add(it.row() + positionInGlobalMatrix[0], it.col() + positionInGlobalMatrix[1], it.value());
            }
        }

    }
}
} // namespace sofa::component::linearsystem
