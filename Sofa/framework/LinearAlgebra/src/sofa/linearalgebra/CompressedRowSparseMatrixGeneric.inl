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
/******************************************************************************
* Contributors:
*   - InSimo
*******************************************************************************/
#pragma once

#include <sofa/linearalgebra/CompressedRowSparseMatrixGeneric.h>

namespace // anonymous
{
    // Boiler-plate code to test if a type implements a method
    // explanation https://stackoverflow.com/a/30848101

    template <typename...>
    using void_t = void;

    // Primary template handles all types not supporting the operation.
    template <typename, template <typename> class, typename = void_t<>>
    struct detectMatrix : std::false_type {};

    // Specialization recognizes/validates only types supporting the archetype.
    template <typename T, template <typename> class Op>
    struct detectMatrix<T, Op, void_t<Op<T>>> : std::true_type {};

    // Actual test if T implements transposed() (hence is a type::Mat)
    template <typename T>
    using isMatrix_t = decltype(std::declval<T>().transposed());

    template <typename T>
    using isMatrix = detectMatrix<T, isMatrix_t>;
} // anonymous

namespace sofa::linearalgebra
{

template<typename TBlock, typename TPolicy>
CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::CompressedRowSparseMatrixGeneric()
    : nBlockRow(0), nBlockCol(0), skipCompressZero(true)
{
}

template<typename TBlock, typename TPolicy>
CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::CompressedRowSparseMatrixGeneric(Index nbBlockRow, Index nbBlockCol)
    : nBlockRow(nbBlockRow), nBlockCol(nbBlockCol)
    , skipCompressZero(true)
{
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::rowBSize() const -> Index
{
    return nBlockRow;
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::colBSize() const -> Index
{
    return nBlockCol;
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::getRowIndex() const -> const VecIndex&
{
    return rowIndex;
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::getRowBegin() const -> const VecIndex&
{
    return rowBegin;
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::getRowRange(Index id) const -> Range
{
    if (id + 1 >= static_cast<Index>(rowBegin.size()))
    {
        return Range(s_invalidIndex, s_invalidIndex);
    }
    return Range(rowBegin[id], rowBegin[id+1]);
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::getColsIndex() const -> const VecIndex&
{
    return colsIndex;
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::getColsValue() const -> const VecBlock&
{
    return colsValue;
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::resizeBlock(Index nbBRow, Index nbBCol)
{
    if (nBlockRow == nbBRow && nBlockRow == nbBCol)
    {
        /// Just clear the matrix
        for (Index i = 0; i < static_cast<Index>(colsValue.size()); ++i)
            traits::clear(colsValue[i]);
        skipCompressZero = colsValue.empty();
        btemp.clear();
    }
    else
    {
        nBlockRow = nbBRow;
        nBlockCol = nbBCol;
        rowIndex.clear();
        rowBegin.clear();
        colsIndex.clear();
        colsValue.clear();
        skipCompressZero = true;
        btemp.clear();
    }
}

template<typename TBlock, typename TPolicy>
bool CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::registerNewCol(Index& colId, TBlock& bvalue)
{
    bool added = false;
    if constexpr (Policy::CompressZeros)
    {
        if (!traits::empty(bvalue))
        {
            colsIndex.push_back(colId);
            colsValue.push_back(bvalue);
            added = true;
        }
    }
    else
    {
        colsIndex.push_back(colId);
        colsValue.push_back(bvalue);
        added = true;
    }
    return added;
}


template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::registerBtempLine(typename VecIndexedBlock::const_iterator& itbtemp) -> std::pair<Index, Index>
{
    Index curentBtempRowID = itbtemp->l;
    Index internalRowBeginCount = 0;
    Index maxColID = std::numeric_limits<Index>::min();
    typename VecIndexedBlock::const_iterator endbtemp = btemp.end();
    while (itbtemp != endbtemp && itbtemp->l == curentBtempRowID)
    {
        Index curentBtempColID = itbtemp->c;
        Block curentBtempValue = itbtemp->value;
        ++itbtemp;
        while (itbtemp != endbtemp && itbtemp->l == curentBtempRowID && itbtemp->c == curentBtempColID)
        {
            curentBtempValue += itbtemp->value;
            ++itbtemp;
        }
        if (registerNewCol(curentBtempColID, curentBtempValue))
        {
            ++internalRowBeginCount;
            if (curentBtempColID > maxColID) maxColID = curentBtempColID;
        }
    }
    return std::make_pair(internalRowBeginCount, maxColID);
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::fullyCompressBtemp()
{
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();

    colsIndex.reserve(btemp.size());
    colsValue.reserve(btemp.size());

    Index rowID = 0;
    Index rowBeginID = 0;
    Index maxColID = std::numeric_limits<Index>::min();

    typename VecIndexedBlock::const_iterator itbtemp  = btemp.begin();
    typename VecIndexedBlock::const_iterator endbtemp = btemp.end();
    while(itbtemp != endbtemp)
    {
        rowID = itbtemp->l;
        rowIndex.push_back(rowID);
        rowBegin.push_back(rowBeginID);
        const auto res = registerBtempLine(itbtemp);
        rowBeginID += res.first;
        if (res.second > maxColID) maxColID = res.second;
    }
    rowBegin.push_back(rowBeginID);
    btemp.clear();

    if constexpr (Policy::AutoSize)
    {
        nBlockRow = rowIndex.back() + 1;
        nBlockCol = maxColID + 1;
    }
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::insertBtemp(const Index i, const Index j) -> Block*
{
    if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
    {
        btemp.push_back(IndexedBlock(i,j));
        traits::clear(btemp.back().value);
    }
    return &btemp.back().value;
}

template<typename TBlock, typename TPolicy>
template< typename >
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::getMaxColIndex() -> Index
{
    Index maxColIndex = 0;
    for (Index rowId = 0; rowId < static_cast<Index>(rowIndex.size()); rowId++)
    {
        Index lastColIndex = colsIndex[rowBegin[rowId+1] - 1];
        if (lastColIndex > maxColIndex) maxColIndex = lastColIndex;
    }
    return maxColIndex;
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::deleteRow(Index rowId)
{
    Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);

    if constexpr(Policy::ClearByZeros)
    {
        for (Index j = rowRange.begin(); j < rowRange.end(); ++j)
        {
            colsValue[j] = Block();
        }
    }
    else
    {
        const std::size_t nnzRow  = std::size_t(rowRange.size());

        colsValue.erase(rowRange.begin(colsValue), rowRange.end(colsValue));
        colsIndex.erase(rowRange.begin(colsIndex), rowRange.end(colsIndex));

        for (std::size_t r = std::size_t(rowId); r < rowBegin.size()-1; ++r)
        {
            rowBegin[r+1] -= nnzRow;
        }
        rowBegin.erase(rowBegin.begin()+rowId);
        rowIndex.erase(rowIndex.begin()+rowId);
        const bool lastRowRemoved = rowIndex.empty();
        nBlockRow = lastRowRemoved ? 0 : rowIndex.back()+1;
        if (lastRowRemoved)
        {
            rowBegin.clear();
            nBlockCol = 0;
        }
        else
        {
            if constexpr(Policy::AutoSize)
            {
                // scan again each row to update nbBlockCol
                nBlockCol = getMaxColIndex()+1;
            }
        }

    }
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::compress()
{
    if (skipCompressZero && btemp.empty())
    {
        return;
    }

    if (!btemp.empty())
    {
        compressBtemp();
    }
    else
    {
        compressCSR();
    }

    skipCompressZero = true;
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::compressBtemp()
{
    std::sort(btemp.begin(), btemp.end());

    /// In This case, matrix is empty, as btemp is sorted just need to fill triplet arrays with btemp
    if (rowIndex.empty())
    {
        fullyCompressBtemp();
        return;
    }

    /// Save actual matrix status
    oldRowIndex.swap(rowIndex);
    oldRowBegin.swap(rowBegin);
    oldColsIndex.swap(colsIndex);
    oldColsValue.swap(colsValue);

    /// New Matrix status with new block added by btemp will be stored here
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();
    touchedBlock.clear();

    rowIndex.reserve(oldRowIndex.size());
    rowBegin.reserve(oldRowIndex.size() + 1);
    colsIndex.reserve(oldColsIndex.size() + btemp.size());
    colsValue.reserve(oldColsValue.size() + btemp.size());

    typename VecIndexedBlock::const_iterator itbtemp  = btemp.begin();
    typename VecIndexedBlock::const_iterator endbtemp = btemp.end();

    /// Info about btemp
    Index curentBtempRowID = itbtemp->l;

    /// Info about old matrix
    Index oldRowIndexCount = 0;
    Index curentOldRowID = oldRowIndex[oldRowIndexCount];
    Index oldNbRow  = Index(oldRowIndex.size());
    Index oldMaxRowID = Index(oldRowIndex.back());

    Index rowBeginCount = 0;
    constexpr Index maxRowID = std::numeric_limits<Index>::max();
    constexpr Index maxColID = std::numeric_limits<Index>::max();

    Index maxRegisteredColID = 0;

    while (itbtemp != endbtemp || curentOldRowID <= oldMaxRowID)
    {
        if (curentOldRowID < curentBtempRowID) /// In this case, we only add old line
        {
            rowIndex.push_back(curentOldRowID);
            rowBegin.push_back(rowBeginCount);
            Range inRow( oldRowBegin[oldRowIndexCount], oldRowBegin[oldRowIndexCount + 1] );
            while (!inRow.empty())
            {
                if (registerNewCol(oldColsIndex[inRow.begin()], oldColsValue[inRow.begin()])) ++rowBeginCount;
                ++inRow;
            }
            ++oldRowIndexCount;
            curentOldRowID = (oldRowIndexCount < oldNbRow ) ? oldRowIndex[oldRowIndexCount] : maxRowID;
        }
        else if (curentOldRowID > curentBtempRowID) /// In this case, we only add btemp line
        {
            rowIndex.push_back(curentBtempRowID);
            rowBegin.push_back(rowBeginCount);
            const auto res = registerBtempLine(itbtemp);
            rowBeginCount += res.first;
            if (res.second > maxRegisteredColID) maxRegisteredColID = res.second;
            curentBtempRowID = (itbtemp != endbtemp) ? itbtemp->l : maxRowID;
        }
        else /// In this case, we add mixed btemp line and old line
        {
            rowIndex.push_back(curentOldRowID);
            rowBegin.push_back(rowBeginCount);
            Range inRow( oldRowBegin[oldRowIndexCount], oldRowBegin[oldRowIndexCount + 1] );
            Index oldColID = (!inRow.empty()) ? oldColsIndex[inRow.begin()] : maxColID;
            Index curentBtempColID = (itbtemp != endbtemp && itbtemp->l == curentBtempRowID) ? itbtemp->c : maxColID;
            while ((itbtemp != endbtemp && itbtemp->l == curentBtempRowID) || !inRow.empty())
            {
                if (oldColID < curentBtempColID) /// In this case, we only add old column
                {
                    if (registerNewCol(oldColID, oldColsValue[inRow.begin()])) ++rowBeginCount;
                    ++inRow;
                    oldColID = (!inRow.empty()) ? oldColsIndex[inRow.begin()] : maxColID;
                }
                else if (oldColID > curentBtempColID) /// In this case, we only add btemp column
                {
                    Block curentBtempValue = itbtemp->value;
                    ++itbtemp;
                    while (itbtemp != endbtemp && itbtemp->l == curentBtempRowID && itbtemp->c == curentBtempColID)
                    {
                        curentBtempValue += itbtemp->value;
                        ++itbtemp;
                    }
                    if (registerNewCol(curentBtempColID, curentBtempValue)) ++rowBeginCount;
                    if (curentBtempColID > maxRegisteredColID) maxRegisteredColID = curentBtempColID;
                    curentBtempColID = (itbtemp != endbtemp && itbtemp->l == curentBtempRowID) ? itbtemp->c : maxColID;
                }
                else
                {
                    Block curentMixedValue = oldColsValue[inRow.begin()];
                    ++inRow;
                    while (itbtemp != endbtemp && itbtemp->l == curentBtempRowID && itbtemp->c == curentBtempColID)
                    {
                        curentMixedValue += itbtemp->value;
                        ++itbtemp;
                    }
                    if (registerNewCol(curentBtempColID, curentMixedValue)) ++rowBeginCount;
                    if (curentBtempColID > maxRegisteredColID) maxRegisteredColID = curentBtempColID;
                    oldColID = (!inRow.empty()) ? oldColsIndex[inRow.begin()] : maxColID;
                    curentBtempColID = (itbtemp != endbtemp && itbtemp->l == curentBtempRowID) ? itbtemp->c : maxColID;
                }
            }
            ++oldRowIndexCount;
            curentBtempRowID = (itbtemp != endbtemp) ? itbtemp->l : maxRowID;
            curentOldRowID = (oldRowIndexCount < oldNbRow ) ? oldRowIndex[oldRowIndexCount] : maxRowID;
        }
    }

    rowBegin.push_back(rowBeginCount);
    btemp.clear();

    if constexpr (Policy::AutoSize)
    {
        nBlockRow = rowIndex.back() + 1;
        if (maxRegisteredColID >= nBlockCol)
        {
            nBlockCol = maxRegisteredColID + 1;
        }
    }
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::compressCSR()
{
    if constexpr (!Policy::CompressZeros) return;
    Index outValues = 0;
    Index outRows = 0;
    for (Index r = 0; r < static_cast<Index>(rowIndex.size()); ++r)
    {
        Index row = rowIndex[r];
        Index rBegin = rowBegin[r];
        Index rEnd = rowBegin[r+1];
        Index outRBegin = outValues;
        for (Index p = rBegin; p != rEnd; ++p)
        {
            if (!traits::empty(colsValue[p]))
            {
                // keep this value
                if (p != outValues)
                {
                    colsValue[outValues] = colsValue[p];
                    colsIndex[outValues] = colsIndex[p];
                }
                ++outValues;
            }
        }
        if(outValues != outRBegin)
        {
            // keep this row
            if (r != outRows)
            {
                rowIndex[outRows] = row;
            }
            if (r != outRows || rBegin != outRBegin)
            {
                rowBegin[outRows] = outRBegin;
            }
            ++outRows;
        }
    }
    if (static_cast<Index>(rowIndex.size()) != outRows || static_cast<Index>(colsIndex.size()) != outValues)
    {
        rowBegin[outRows] = outValues;
        rowIndex.resize(outRows);
        rowBegin.resize(outRows+1);
        colsIndex.resize(outValues);
        colsValue.resize(outValues);
    }
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::swap(Matrix& m)
{
    Index t;
    t = nBlockRow; nBlockRow = m.nBlockRow; m.nBlockRow = t;
    t = nBlockCol; nBlockCol = m.nBlockCol; m.nBlockCol = t;
    bool b;
    b = skipCompressZero; skipCompressZero = m.skipCompressZero; m.skipCompressZero = b;
    rowIndex.swap(m.rowIndex);
    rowBegin.swap(m.rowBegin);
    colsIndex.swap(m.colsIndex);
    colsValue.swap(m.colsValue);
    btemp.swap(m.btemp);
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::fullRows()
{
    if constexpr (Policy::AutoCompress) compress();
    if (static_cast<Index>(rowIndex.size()) >= nBlockRow) return;
    oldRowIndex.swap(rowIndex);
    oldRowBegin.swap(rowBegin);
    rowIndex.resize(nBlockRow);
    rowBegin.resize(nBlockRow+1);
    for (Index i=0; i<nBlockRow; ++i)
        rowIndex[i] = i;
    Index j = 0;
    Index b = 0;
    for (Index i = 0; i < static_cast<Index>(oldRowIndex.size()); ++i)
    {
        b = oldRowBegin[i];
        for (; j<=oldRowIndex[i]; ++j)
            rowBegin[j] = b;
    }
    b = !oldRowBegin.empty() ? oldRowBegin[oldRowBegin.size()-1] : Index(0);
    for (; j<=nBlockRow; ++j)
        rowBegin[j] = b;
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::shiftIndices(Index base)
{
    for (Index i=0; i<(Index)rowIndex.size(); ++i)
        rowIndex[i] += base;
    for (Index i=0; i<(Index)rowBegin.size(); ++i)
        rowBegin[i] += base;
    for (Index i=0; i<(Index)colsIndex.size(); ++i)
        colsIndex[i] += base;
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::block(Index i, Index j) const -> const Block&
{
    static Block empty;

    /// \warning this violates the const-ness of the method !
    /// But if AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.
    if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress();

    if (rowIndex.empty() || i > rowIndex.back()) return empty; /// Matrix is empty or index is upper than registered lines
    if constexpr (Policy::AutoSize) if (j > nBlockCol) return empty; /// Matrix is auto sized so requested column could not exist

    Index rowId = 0;
    if (i == rowIndex.back()) rowId = Index(rowIndex.size() - 1); /// Optimization to avoid do a find when looking for the last line registred
    else if (i == rowIndex.front()) rowId = 0;             /// Optimization to avoid do a find when looking for the first line registred
    else
    {
        rowId = (nBlockRow == 0) ? 0 : Index(i * rowIndex.size() / nBlockRow);
        if (!sortedFind(rowIndex, i, rowId)) return empty;
    }

    Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
    Index colId = 0;
    if (j == colsIndex[rowRange.first]) colId = rowRange.first;                /// Optimization to avoid do a find when looking for the first column registred for specific column
    else if (j == colsIndex[rowRange.second - 1]) colId = rowRange.second - 1; /// Optimization to avoid do a find when looking for the last column registred for specific column
    else
    {
        colId = (nBlockCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlockCol;
        if (!sortedFind(colsIndex, rowRange, j, colId)) return empty;
    }

    return colsValue[colId];
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::wblock(Index i, Index j, bool create) -> Block*
{
    if constexpr (Policy::OrderedInsertion)
    {
        /// Matrix is empty or index is upper than registered lines
        if (rowIndex.empty() || i > rowIndex.back()) /// Optimization we are registering value at end
        {
            if (!create) return nullptr;
            if (rowIndex.empty() && rowBegin.empty()) rowBegin.push_back(0);
            rowIndex.push_back(i);
            colsIndex.push_back(j);
            colsValue.push_back(Block());
            rowBegin.push_back(Index(colsIndex.size()));

            if constexpr (Policy::AutoSize)
            {
                nBlockRow = i + 1;
                if (j > nBlockCol) nBlockCol = j + 1;
            }
            return &colsValue.back();
        }
        else if (i == rowIndex.back()) /// In this case, we are trying to write on last registered line
        {
            Index rowId = Index(rowIndex.size() - 1);
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            if (j == colsIndex[rowRange.second - 1]) /// In this case, we are trying to write on last registered column, directly return ref on it
            {
                return &colsValue[rowRange.second - 1];
            }
            else if (j > colsIndex[rowRange.second - 1]) /// Optimization we are trying to write on last line et upper of last column, directly create it.
            {
                if (!create) return nullptr;
                colsIndex.push_back(j);
                colsValue.push_back(Block());
                rowBegin.back()++;
                if constexpr (Policy::AutoSize)
                {
                    if (j > nBlockCol) nBlockCol = j + 1;
                }
                return &colsValue.back();
            }
            else
            {
                Index colId = (nBlockCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlockCol;
                if (!sortedFind(colsIndex, rowRange, j, colId)) return create ? insertBtemp(i,j) : nullptr;
                return &colsValue[colId];
            }
        }

        if constexpr (Policy::AutoSize) if (j > nBlockCol) return create ? insertBtemp(i,j) : nullptr; /// Matrix is auto sized so requested column could not exist

        Index rowId = 0;
        if (i == rowIndex.back()) rowId = Index(rowIndex.size() - 1);      /// Optimization to avoid do a find when looking for the last line registred
        else if (i == rowIndex.front()) rowId = 0;                  /// Optimization to avoid do a find when looking for the first line registred
        else
        {
            rowId = (nBlockRow == 0) ? 0 : Index(i * rowIndex.size() / nBlockRow);
            if (!sortedFind(rowIndex, i, rowId)) return create ? insertBtemp(i,j) : nullptr;
        }

        Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
        Index colId = 0;
        if (j == colsIndex[rowRange.first]) colId = rowRange.first;                /// Optimization to avoid do a find when looking for the first column registred for specific column
        else if (j == colsIndex[rowRange.second - 1]) colId = rowRange.second - 1; /// Optimization to avoid do a find when looking for the last column registred for specific column
        else
        {
            colId = (nBlockCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlockCol;
            if (!sortedFind(colsIndex, rowRange, j, colId)) return create ? insertBtemp(i,j) : nullptr;
        }

        return &colsValue[colId];
    }
    else
    {
        Index rowId = (nBlockRow == 0) ? 0 : Index(i * rowIndex.size() / nBlockRow);
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            Index colId = (nBlockCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlockCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {
                return &colsValue[colId];
            }
        }
        if (create)
        {
            if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
            {
                btemp.push_back(IndexedBlock(i,j));
                traits::clear(btemp.back().value);
            }
            return &btemp.back().value;
        }
        return nullptr;
    }
}


template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::wblock(Index i, Index j, Index& rowId, Index& colId, bool create) -> Block*
{
    bool rowFound = true;
    if (rowId < 0 || rowId >= static_cast<Index>(rowIndex.size()) || rowIndex[rowId] != i)
    {
        rowId = Index(i * rowIndex.size() / nBlockRow);
        rowFound = sortedFind(rowIndex, i, rowId);
    }
    if (rowFound)
    {
        bool colFound = true;
        Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
        if (colId < rowRange.begin() || colId >= rowRange.end() || colsIndex[colId] != j)
        {
            colId = rowRange.begin() + j * rowRange.size() / nBlockCol;
            colFound = sortedFind(colsIndex, rowRange, j, colId);
        }
        if (colFound)
        {
            return &colsValue[colId];
        }
    }

    if (create)
    {
        if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
        {
            btemp.push_back(IndexedBlock(i,j));
            traits::clear(btemp.back().value);
        }
        return &btemp.back().value;
    }
    return nullptr;
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::getBlock(Index i, Index j) const -> const Block&
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (i > j) assert(false);
    return block(i,j);
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::getSymBlock(Index i, Index j) const -> const BlockTranspose
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (i > j) return traits::transposed( block(i,j) );
    return getBlock(i,j);
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::setBlock(Index i, Index j, const Block& v)
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (i > j) return;
    *wblock(i,j,true) = v;
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addBlock(Index i, Index j, const Block& v)
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (i > j) return;
    *wblock(i,j,true) += v;
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::setBlock(Index i, Index j, Index& rowId, Index& colId, const Block& v)
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (i > j) return;
    *wblock(i,j,rowId,colId,true) = v;
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addBlock(Index i, Index j, Index& rowId, Index& colId, const Block& v)
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (i > j) return;
    *wblock(i,j,rowId,colId,true) += v;
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::getWBlock(Index i, Index j, bool create) -> Block*
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (i > j) return nullptr;
    return wblock(i,j,create);
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::clearRowBlock(Index i)
{
    if constexpr (Policy::AutoCompress) compress(); /// If AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.
    if constexpr (Policy::IsAlwaysSquare && !Policy::ClearByZeros) /// In Square case removing only Row will produce not quare matrix
    {
        clearRowColBlock(i);
        return;
    }

    Index rowId = 0;
    if (i == rowIndex.back()) rowId = Index(rowIndex.size() - 1);      /// Optimization to avoid do a find when looking for the last line registred
    else if (i == rowIndex.front()) rowId = 0;                  /// Optimization to avoid do a find when looking for the first line registred
    else
    {
        rowId = (nBlockRow == 0) ? 0 : Index(i * rowIndex.size() / nBlockRow);
        if (!sortedFind(rowIndex, i, rowId)) return;
    }

    deleteRow(rowId);

    if constexpr (Policy::AutoCompress && Policy::ClearByZeros) compress(); /// If AutoCompress policy is activated, need to compress zeros.
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::clearColBlock(Index j)
{
    /// If AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.
    if constexpr (Policy::AutoCompress) compress();
    if constexpr (Policy::IsAlwaysSquare && !Policy::ClearByZeros) /// In Square case removing only Col will produce not square matrix
    {
        clearRowColBlock(j);
        return;
    }

    for (Index rowId = static_cast<Index>(rowIndex.size())-1; rowId >=0 ; --rowId)
    {
        Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);

        Index colId = -1;
        if (j == colsIndex[rowRange.first]) colId = rowRange.first;                /// Optimization to avoid do a find when looking for the first column registred for specific column
        else if (j == colsIndex[rowRange.second - 1]) colId = rowRange.second - 1; /// Optimization to avoid do a find when looking for the last column registred for specific column
        else
        {
            colId = (nBlockCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlockCol;
            if (!sortedFind(colsIndex, rowRange, j, colId)) colId = -1;
        }
        if (colId != -1) /// Means col exist in this line
        {
            if constexpr (Policy::ClearByZeros)
            {
                colsValue[colId] = Block();
            }
            else
            {
                if constexpr (Policy::AutoCompress)
                {
                    /// In this case, line was containing only this column, directly clearing is faster than putting to zero and compressing.
                    if (rowRange.second - 1 == rowRange.first)
                    {
                        deleteRow(rowId);
                        continue;
                    }
                }

                for (auto it = std::next(rowBegin.begin(), rowId + 1); it != rowBegin.end(); it++)
                    *it -= 1;

                colsIndex.erase(std::next(colsIndex.begin(), colId));
                colsValue.erase(std::next(colsValue.begin(), colId));
            }
        }
    }

    if constexpr (Policy::AutoSize)
    {
        nBlockRow = rowIndex.empty() ? 0 : rowIndex.back() + 1; /// To be sure if row has been erased
        if (j == nBlockCol - 1) nBlockCol = getMaxColIndex() + 1;
    }

    if constexpr (Policy::AutoCompress && Policy::ClearByZeros) compress(); /// If AutoCompress policy is activated, need to compress zeros.
}

template<typename TBlock, typename TPolicy>
auto CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::countEmptyBlocks() const -> std::size_t
{
    return std::count_if(this->colsValue.cbegin(), this->colsValue.cend(), [] (const Block& b)
    {
        return traits::empty(b);
    });
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::clearRowColBlock(Index i)
{
    /// If AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.
    if constexpr (Policy::AutoCompress) compress();

    bool foundRowId = true;
    Index rowId = 0;
    if (i == rowIndex.back()) rowId = rowIndex.size() - 1;      /// Optimization to avoid do a find when looking for the last line registred
    else if (i == rowIndex.front()) rowId = 0;                  /// Optimization to avoid do a find when looking for the first line registred
    else
    {
        rowId = (nBlockRow == 0) ? 0 : i * rowIndex.size() / nBlockRow;
        if (!sortedFind(rowIndex, i, rowId)) foundRowId = false;
    }

    bool foundColId = true;
    Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
    Index colId = 0;
    if (i == colsIndex[rowRange.first]) colId = rowRange.first;                /// Optimization to avoid do a find when looking for the first column registred for specific column
    else if (i == colsIndex[rowRange.second - 1]) colId = rowRange.second - 1; /// Optimization to avoid do a find when looking for the last column registred for specific column
    else
    {
        colId = (nBlockCol == 0) ? 0 : rowRange.begin() + i * rowRange.size() / nBlockCol;
        if (!sortedFind(colsIndex, rowRange, i, colId)) foundColId = false;;
    }

    if (!foundRowId && !foundColId)
    {
        msg_error("CompressedRowSparseMatrixGeneric") << "invalid write access to row and column "<<i<<" in "<< this->Name() << " of size ("<<rowBSize()<<","<<colBSize()<<")";
        return;
    }

    deleteRow(rowId); /// Do not call clearRow to only compress zero if activated once.
    clearColBlock(i);
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::clear()
{
    if constexpr (Policy::ClearByZeros)
    {
        for (Index i = 0; i < static_cast<Index>(colsValue.size()); ++i)
            traits::clear(colsValue[i]);
        skipCompressZero = colsValue.empty();
    }
    else
    {
        rowIndex.clear();
        rowBegin.clear();
        colsIndex.clear();
        colsValue.clear();
        nBlockRow = 0;
        nBlockCol = 0;
        skipCompressZero = true;
    }

    btemp.clear();
    compress();
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::add(unsigned int bi, unsigned int bj, const Block& b)
{
    addBlock(bi, bj, b);
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::add(unsigned int bi, unsigned int bj, int& rowId, int& colId, const Block& b)
{
    addBlock(bi, bj, rowId, colId, b);
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addDBlock(unsigned int bi, unsigned int bj, const DBlock& b)
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (bi > bj) return;
    Block* mb = wblock(bi, bj, true);

    for (unsigned int i = 0; i < NL; ++i)
        traits::vadd(*mb, i, i, b[i] );
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addDValue(unsigned int bi, unsigned int bj, const Real b)
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (bi > bj) return;
    Block* mb = wblock(bi, bj, true);

    for (unsigned int i = 0; i < NL; ++i)
        traits::vadd(*mb, i, i, b);
}

template<typename TBlock, typename TPolicy>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addDValue(unsigned int bi, unsigned int bj, int& rowId, int& colId, const Real b)
{
    if constexpr (!Policy::StoreLowerTriangularBlock) if (bi > bj) return;
    Block* mb = wblock(bi, bj, rowId, colId, true);

    for (unsigned int i = 0; i < NL; ++i)
        traits::vadd(*mb, i, i, b);
}


template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addDiag(unsigned int bi, const Block& b)
{
    add(bi, bi, b);
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addDiag(unsigned int bi, int& rowId, int& colId, const Block &b)
{
    add(bi, bi, rowId, colId, b);
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addDiagDBlock(unsigned int bi, const DBlock& b)
{
    addDBlock(bi, bi, b);
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addDiagDValue(unsigned int bi, const Real b)
{
    addDValue(bi, bi, b);
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addDiagDValue(unsigned int bi, int& rowId, int& colId, const Real b)
{
    addDValue(bi, bi, rowId, colId, b);
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addSym(unsigned int bi, unsigned int bj, const Block& b)
{
    if constexpr(Policy::StoreLowerTriangularBlock)
    {
        add(bi, bj, b);
        add(bj, bi, traits::transposed(b) );
    }
    else
    {
        if (bi > bj) // the block we received is in the lower triangular
        {
            add(bj, bi, traits::transposed(b) );
        }
        else
        {
            add(bi, bj, b);
        }
    }
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addSym(unsigned int bi, unsigned int bj, int& rowId, int& colId, int& rowIdT, int& colIdT, const Block &b)
{
    if constexpr(Policy::StoreLowerTriangularBlock)
    {
        add(bi, bj, rowId, colId, b);
        add(bj, bi, rowIdT, colIdT, traits::transposed(b) );
    }
    else
    {
        if (bi > bj) // the block we received is in the lower triangular
        {
            add(bj, bi, rowIdT, colIdT, traits::transposed(b) );
        }
        else
        {
            add(bi, bj, rowId, colId, b);
        }
    }
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addSymDBlock(unsigned int bi, unsigned int bj, const DBlock& b)
{
    const unsigned int i = std::min(bi, bj);
    const unsigned int j = std::max(bi, bj);
    addDBlock(i, j, b);
    if constexpr (Policy::StoreLowerTriangularBlock) addDBlock(j, i, b);
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addSymDValue(unsigned int bi, unsigned int bj, const Real b)
{
    const unsigned int i = std::min(bi, bj);
    const unsigned int j = std::max(bi, bj);
    addDValue(i, j, b);
    if constexpr (Policy::StoreLowerTriangularBlock) addDValue(j, i, b);
}

template<typename TBlock, typename TPolicy>
template< typename >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::addSymDValue(unsigned int bi, unsigned int bj, int& rowId, int& colId, int& rowIdT, int& colIdT, Real b)
{
    const unsigned int i = std::min(bi, bj);
    const unsigned int j = std::max(bi, bj);
    addDValue(i, j, rowId, colId, b);
    if constexpr (Policy::StoreLowerTriangularBlock) addDValue(j, i, rowIdT, colIdT, b);
}

template<typename TBlock, typename TPolicy>
template<typename TBlock2, typename TPolicy2>
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::transposeFullRows(CompressedRowSparseMatrixGeneric<TBlock2, TPolicy2>& res) const
{
    res.nBlockCol = nBlockRow;
    res.nBlockRow = nBlockCol;

    res.rowBegin.clear();
    res.rowBegin.resize(res.nBlockRow+1,0);

    res.colsIndex.clear();
    res.colsIndex.resize(this->colsIndex.size(),0);

    res.colsValue.clear();
    res.colsValue.resize(this->colsValue.size());

    res.rowIndex.clear();

    for (unsigned i = 0; i<rowIndex.size(); ++i)
    {
        for (int p = rowBegin[i]; p<rowBegin[i+1]; ++p)
        {
            ++res.rowBegin[colsIndex[p]];
        }
    }

    Index count = 0;
    VecIndex positions(res.nBlockRow);

    for (int i=0; i<res.nBlockRow; ++i)
    {
        Index tmp = res.rowBegin[i];
        res.rowBegin[i] = count;
        positions[i] = count;
        count += tmp;
    }
    res.rowBegin[ res.nBlockRow ] = count;

    for (unsigned i=0; i<rowIndex.size(); ++i)
    {
        int row = rowIndex[i];

        for (int p=rowBegin[i]; p<rowBegin[i+1]; ++p)
        {
            Index col                 = colsIndex[p];
            Index pos                 = positions[col];
            res.colsIndex[pos]        = row;
            res.colsValue[pos]        = colsValue[p];
            ++positions[col];
         }
    }

    res.rowIndex.resize(res.rowBegin.size()-1);
    for (unsigned i=0; i<res.rowIndex.size(); ++i)
    {
        res.rowIndex[i] = i;
    }
}

template<typename TBlock, typename TPolicy>
template<typename RB, typename RP, typename MB, typename MP >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::mul( CompressedRowSparseMatrixGeneric<RB,RP>& res, const CompressedRowSparseMatrixGeneric<MB,MP>& m ) const
{
    if constexpr (!std::is_arithmetic_v<Block> && !std::is_arithmetic_v<RB> && !std::is_arithmetic_v<MB>)
    {
        assert(Block::nbCols == MB::nbLines);
        assert(RB::nbLines == Block::nbLines);
        assert(MB::nbCols == RB::nbCols);
    }

    assert( colBSize() == m.rowBSize() );

    if constexpr (Policy::AutoCompress)
    {
        const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        (const_cast<CompressedRowSparseMatrixGeneric<MB,MP>*>(&m))->compress();  /// \warning this violates the const-ness of the parameter
    }

    res.resizeBlock( this->nBlockRow, m.nBlockCol );  // clear and resize the result

    if( m.rowIndex.empty() ) return; // if m is null

    for( Index xi = 0; xi < Index(rowIndex.size()); ++xi )  // for each non-null block row
    {
        unsigned mr = 0; // block row index in m

        Index row = rowIndex[xi];      // block row

        Range rowRange( rowBegin[xi], rowBegin[xi+1] );
        for( Index xj = rowRange.begin() ; xj < rowRange.end() ; ++xj )  // for each non-null block
        {
            Index col = colsIndex[xj];     // block column
            const Block& b = colsValue[xj]; // block value

            // find the non-null row in m, if any
            while( mr<m.rowIndex.size() && m.rowIndex[mr]<col ) mr++;
            if( mr==m.rowIndex.size() || m.rowIndex[mr] > col ) continue;  // no matching row, ignore this block

            // Accumulate  res[row] += b * m[col]
            Range mrowRange( m.rowBegin[mr], m.rowBegin[mr+1] );
            for( Index mj = mrowRange.begin() ; mj< mrowRange.end() ; ++mj ) // for each non-null block in  m[col]
            {
                Index mcol = m.colsIndex[mj];     // column index of the non-null block
                *res.wblock(row,mcol,true) += b * m.colsValue[mj];  // find the matching block in res, and accumulate the block product
            }
        }
    }
    res.compress();
}

template<typename TBlock, typename TPolicy>
TBlock CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::blockMultTranspose(const TBlock& blockA, const TBlock& blockB)
{
    if constexpr (isMatrix<Block>())
    {
        return blockA.multTranspose(blockB);
    }
    else
    {
        return blockA * blockB;
    }
}

template<typename TBlock, typename TPolicy>
template<typename RB, typename RP, typename MB, typename MP >
void CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::mulTranspose( CompressedRowSparseMatrixGeneric<RB,RP>& res, const CompressedRowSparseMatrixGeneric<MB,MP>& m ) const
{
    if constexpr (!std::is_arithmetic_v<Block> && !std::is_arithmetic_v<RB> && !std::is_arithmetic_v<MB>)
    {
        assert(Block::nbLines == MB::nbLines);
        assert(RB::nbLines == Block::nbCols);
        assert(MB::nbCols == RB::nbCols);
    }

    assert( rowBSize() == m.rowBSize() );

    if constexpr (Policy::AutoCompress)
    {
        const_cast<Matrix*>(this)->compress();  /// \warning this violates the const-ness of the method
        (const_cast<CompressedRowSparseMatrixGeneric<MB,MP>*>(&m))->compress();  /// \warning this violates the const-ness of the parameter
    }


    res.resizeBlock( this->nBlockCol, m.nBlockCol );  // clear and resize the result

    if( m.rowIndex.empty() ) return; // if m is null

    for( Size xi = 0 ; xi < rowIndex.size() ; ++xi )  // for each non-null transpose block column
    {
        unsigned mr = 0; // block row index in m

        Index col = rowIndex[xi];      // block col (transposed col = row)

        Range rowRange( rowBegin[xi], rowBegin[xi+1] );
        for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)  // for each non-null block
        {
            Index row = colsIndex[xj];     // block row (transposed row = col)
            const Block& b = colsValue[xj]; // block value

            // find the non-null row in m, if any
            while( mr<m.rowIndex.size() && m.rowIndex[mr]<col ) mr++;
            if( mr==m.rowIndex.size() || m.rowIndex[mr] > col ) continue;  // no matching row, ignore this block

            // Accumulate  res[row] += b^T * m[col]
            Range mrowRange( m.rowBegin[mr], m.rowBegin[mr+1] );
            for( Index mj = mrowRange.begin() ; mj< mrowRange.end() ; ++mj ) // for each non-null block in  m[col]
            {
                Index mcol = m.colsIndex[mj];     // column index of the non-null block
                //*res.wblock(row,mcol,true) += b.multTranspose( m.colsValue[mj] );  // find the matching block in res, and accumulate the block product
                *res.wblock(row, mcol, true) += blockMultTranspose(b, m.colsValue[mj]);  // find the matching block in res, and accumulate the block product
            }
        }
    }
    res.compress();
}

template<typename TBlock, typename TPolicy>
bool CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::check_matrix()
{
    return check_matrix(
            Index(this->getColsValue().size()),
            this->rowBSize(),
            this->colBSize(),
            static_cast<Index*> (&(rowBegin[0])),
            static_cast<Index*> (&(colsIndex[0])),
            static_cast<Block*> (&(colsValue[0]))
            );
}

template<typename TBlock, typename TPolicy>
bool CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::check_matrix(
    Index nzmax,    // nb values
    Index m,        // number of row
    Index n,        // number of columns
    Index * a_p,    // column pointers (size n+1) or col indices (size nzmax)
    Index * a_i,    // row indices, size nzmax
    Block * a_x      // numerical values, size nzmax
)
{
    // check ap, size m beecause ther is at least the diagonal value wich is different of 0
    if (a_p[0]!=0)
    {
        msg_error("CompressedRowSparseMatrixGeneric") << "First value of row indices (a_p) should be 0";
        return false;
    }

    for (Index i=1; i<=m; i++)
    {
        if (a_p[i]<=a_p[i-1])
        {
            msg_error("CompressedRowSparseMatrixGeneric") << "Row (a_p) indices are not sorted index " << i-1 << " : " << a_p[i-1] << " , " << i << " : " << a_p[i];
            return false;
        }
    }
    if (nzmax == -1)
    {
        nzmax = a_p[m];
    }
    else if (a_p[m]!=nzmax)
    {
        msg_error("CompressedRowSparseMatrixGeneric") << "Last value of row indices (a_p) should be " << nzmax << " and is " << a_p[m];
        return false;
    }


    Index k=1;
    for (Index i=0; i<nzmax; i++)
    {
        i++;
        for (; i<a_p[k]; i++)
        {
            if (a_i[i] <= a_i[i-1])
            {
                msg_error("CompressedRowSparseMatrixGeneric") << "Column (a_i) indices are not sorted index " << i-1 << " : " << a_i[i-1] << " , " << i << " : " << a_p[i];
                return false;
            }
            if (a_i[i]<0 || a_i[i]>=n)
            {
                msg_error("CompressedRowSparseMatrixGeneric") << "Column (a_i) indices are not correct " << i << " : " << a_i[i];
                return false;
            }
        }
        k++;
    }

    for (Index i=0; i<nzmax; i++)
    {
        if (traits::empty(a_x[i]))
        {
            msg_error("CompressedRowSparseMatrixGeneric") << "Warning, matrix contains empty block at index " << i;
            return false;
        }
    }

    if (n!=m)
    {
        msg_error("CompressedRowSparseMatrixGeneric") << "the matrix is not square";
        return false;
    }

    msg_error("CompressedRowSparseMatrixGeneric") << "Check_matrix passed successfully";
    return true;
}

} // namespace sofa::linearalgebra
