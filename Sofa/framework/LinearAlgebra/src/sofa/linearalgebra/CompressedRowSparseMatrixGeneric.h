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

#include <sofa/linearalgebra/config.h>

#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/linearalgebra/MatrixExpr.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/matrix_bloc_traits.h>
#include <algorithm>

namespace sofa::linearalgebra
{

/// Traits class which defines the containers to use for a given type of block
template<class Block>
struct CRSBlockTraits
{
    using VecBlock  = sofa::type::vector<Block>;
    using VecIndex = sofa::type::vector<sofa::SignedIndex>;
    using VecFlag  = sofa::type::vector<bool>;
};

struct CRSDefaultPolicy
{
    /// Set to true if this matrix is always square (must be true for symmetric)
    static constexpr bool IsAlwaysSquare = false;
    /// Set to true if this matrix is always symmetric (IsAlwaysSquare should be true)
    static constexpr bool IsAlwaysSymmetric = false;
    /// Set to true if the size of the matrix should be automatically increased when new blocks are added
    static constexpr bool AutoSize = false;
    /// Set to true if the matrix should be automatically compressed (easier to use, but might cause issues in multithreading)
    static constexpr bool AutoCompress = true;
    /// Set to true if the blocks that are all zeros should be removed from the matrix when compressing (expensive)
    static constexpr bool CompressZeros = true;
    /// Set to true if clear methods will put all concerned value to zero instead of clearing vectors (CompressZeros should be true)
    static constexpr bool ClearByZeros = true;
    /// Set to true if insertion in matrix are in most case at last line index or last col index
    static constexpr bool OrderedInsertion = false;
    /// Set to false to disable storage of blocks on the lower triangular part (IsAlwaysSymmetric must be true)
    static constexpr bool StoreLowerTriangularBlock = true;

    /// Do not change this value, has to be overrided for all derivated class
    static constexpr int matrixType = 0;
};

template<typename TBlock, typename TPolicy = CRSDefaultPolicy>
class CompressedRowSparseMatrixGeneric : public TPolicy
{
public:
    typedef CompressedRowSparseMatrixGeneric<TBlock,TPolicy> Matrix;

    typedef TBlock Block;
    typedef TPolicy Policy;
    typedef matrix_bloc_traits<Block, sofa::SignedIndex> traits;
    typedef typename traits::BlockTranspose BlockTranspose;
    typedef typename traits::Real Real;

    typedef Matrix Expr;
    enum { category = MATRIX_SPARSE };
    enum { operand = 1 };

    static constexpr sofa::Index NL = traits::NL;  ///< Number of rows of a block
    static constexpr sofa::Index NC = traits::NC;  ///< Number of columns of a block

    using VecBlock  = typename CRSBlockTraits<Block>::VecBlock;
    using VecIndex = typename CRSBlockTraits<Block>::VecIndex;
    using VecFlag  = typename CRSBlockTraits<Block>::VecFlag;
    using Index = typename VecIndex::value_type;
    static constexpr Index s_invalidIndex = std::is_signed_v<Index> ? std::numeric_limits<Index>::lowest() : std::numeric_limits<Index>::max();

    typedef sofa::type::Vec<NC,Real> DBlock;

    static_assert(!(Policy::IsAlwaysSymmetric && !Policy::IsAlwaysSquare),
        "IsAlwaysSymmetric can only be true if IsAlwaysSquare is true");
    static_assert(!(!Policy::StoreLowerTriangularBlock && !Policy::IsAlwaysSymmetric),
        "StoreLowerTriangularBlock can only be false if IsAlwaysSymmetric is true");

    struct IndexedBlock
    {
        Index l,c;
        Block value;
        IndexedBlock() {}
        IndexedBlock(Index i, Index j) : l(i), c(j) {}
        IndexedBlock(Index i, Index j, const Block& v) : l(i), c(j), value(v) {}
        bool operator < (const IndexedBlock& b) const
        {
            return (l < b.l) || (l == b.l && c < b.c);
        }
        bool operator <= (const IndexedBlock& b) const
        {
            return (l < b.l) || (l == b.l && c <= b.c);
        }
        bool operator > (const IndexedBlock& b) const
        {
            return (l > b.l) || (l == b.l && c > b.c);
        }
        bool operator >= (const IndexedBlock& b) const
        {
            return (l > b.l) || (l == b.l && c >= b.c);
        }
        bool operator == (const IndexedBlock& b) const
        {
            return (l == b.l) && (c == b.c);
        }
        bool operator != (const IndexedBlock& b) const
        {
            return (l != b.l) || (c != b.c);
        }
    };
    typedef type::vector<IndexedBlock> VecIndexedBlock;

    class Range : public std::pair<Index, Index>
    {
        typedef std::pair<Index, Index> Inherit;
    public:
        Range() : Inherit(0,0) {}
        Range(Index begin, Index end) : Inherit(begin,end) {}
        Index begin() const { return this->first; }
        Index end() const { return this->second; }
        void setBegin(Index i) { this->first = i; }
        void setEnd(Index i) { this->second = i; }
        bool empty() const { return begin() == end(); }
        Index size() const { return end()-begin(); }
        typename VecBlock::iterator begin(VecBlock& b) const { return b.begin() + begin(); }
        typename VecBlock::iterator end  (VecBlock& b) const { return b.begin() + end  (); }
        typename VecBlock::const_iterator begin(const VecBlock& b) const { return b.begin() + begin(); }
        typename VecBlock::const_iterator end  (const VecBlock& b) const { return b.begin() + end  (); }
        typename VecIndex::iterator begin(VecIndex& b) const { return b.begin() + begin(); }
        typename VecIndex::iterator end  (VecIndex& b) const { return b.begin() + end  (); }
        typename VecIndex::const_iterator begin(const VecIndex& b) const { return b.begin() + begin(); }
        typename VecIndex::const_iterator end  (const VecIndex& b) const { return b.begin() + end  (); }
        void operator++() { ++this->first; }
        void operator++(int) { ++this->first; }
        bool isInvalid() const
        {
            return this->first == s_invalidIndex || this->second == s_invalidIndex;
        }
    };

    static bool sortedFind(const VecIndex& v, Range in, Index val, Index& result)
    {
        if (in.empty()) return false;
        Index candidate = (result >= in.begin() && result < in.end()) ? result : ((in.begin() + in.end()) >> 1);
        for(;;)
        {
            Index i = v[candidate];
            if (i == val) { result = candidate; return true; }
            if (i < val)  in.setBegin(candidate+1);
            else          in.setEnd(candidate);
            if (in.empty()) break;
            candidate = (in.begin() + in.end()) >> 1;
        }
        return false;
    }

    static bool sortedFind(const VecIndex& v, Index val, Index& result)
    {
        return sortedFind(v, Range(0, Index(v.size())), val, result);
    }

public :
    /// Size
    Index nBlockRow,nBlockCol; ///< Mathematical size of the matrix, in blocks.

    /// Compressed sparse data structure
    VecIndex rowIndex;    ///< indices of non-empty block rows
    VecIndex rowBegin;    ///< column indices of non-empty blocks in each row. The column indices of the non-empty block within the i-th non-empty row are all the colsIndex[j],  j  in [rowBegin[i],rowBegin[i+1])
    VecIndex colsIndex;   ///< column indices of all the non-empty blocks, sorted by increasing row index and column index
    VecBlock colsValue;   ///< values of the non-empty blocks, in the same order as in colsIndex
    VecFlag  touchedBlock; ///< boolean vector, i-th value is true if block has been touched since last compression.

    /// Additional storage to make block insertion more efficient
    VecIndexedBlock btemp; ///< unsorted blocks and their indices

    /// When true, only compressBtemp if needed
    /// This is to avoid compressCRS costly method when no change into matrix size occurs.
    bool skipCompressZero;

    /// Temporary vectors used during compression
    VecIndex oldRowIndex;
    VecIndex oldRowBegin;
    VecIndex oldColsIndex;
    VecBlock  oldColsValue;

    CompressedRowSparseMatrixGeneric();

    CompressedRowSparseMatrixGeneric(Index nbBlockRow, Index nbBlockCol);

    virtual ~CompressedRowSparseMatrixGeneric() = default;

    /// \returns the number of row blocks
    Index rowBSize() const;

    /// \returns the number of col blocks
    Index colBSize() const;

    const VecIndex& getRowIndex() const;
    const VecIndex& getRowBegin() const;

    /// Returns the range of indices from the column indices corresponding to the id-th row
    Range getRowRange(Index id) const;

    const VecIndex& getColsIndex() const;
    const VecBlock& getColsValue() const;

    virtual void resizeBlock(Index nbBRow, Index nbBCol);

    SOFA_ATTRIBUTE_DISABLED__CRS_BLOCK_RENAMING()
    void resizeBloc(Index nbBRow, Index nbBCol)
    {
        resizeBlock(nbBRow, nbBCol);
    }

protected:

    /**
    * \brief Add a new col into matrix
    * @param colId : Index of column
    * @param bvalue : Block value to add
    * @return true if col has been added
    **/
    bool registerNewCol(Index& colId, TBlock& bvalue);

    /**
    * \brief Add a complete new line from btemp into matrix
    * @param itbtemp : Reference to actual status of iterator on btemp
    * @return Number of col added
    **/
    std::pair<Index, Index> registerBtempLine(typename VecIndexedBlock::const_iterator& itbtemp);

    /**
    * \brief Clear matrix and just add btemp array
    **/
    void fullyCompressBtemp();

    /**
    * \brief Method to easy insert new block into btemp.
    * @param Line index i and column index j
    * @return pointer on Block value
    **/
    Block* insertBtemp(const Index i, const Index j);

    /**
    * \brief Method to easy have the max colIndex.
    * Could only be used if AutoSize policy is activated.
    **/
    template< typename = typename std::enable_if< Policy::AutoSize> >
    Index getMaxColIndex();

    /**
    * \brief Method to easy delete row given position in rowIndex.
    * @param RowId position on line in rowIndex
    **/
    void deleteRow(Index rowId);

public:

    void compress();

protected:
    /**
    * \brief Clear matrix and compute new triplet's arrays by combining old ones and btemp(VecIndexedBlock) array
    **/
    void compressBtemp();

    void compressCSR();

public:

    void swap(Matrix& m);

    /// Make sure all rows have an entry even if they are empty
    void fullRows();

    /// Add the given base to all indices.
    /// Use 1 to convert do Fortran 1-based notation.
    /// Note that the matrix will no longer be valid
    /// from the point of view of C/C++ codes. You need
    /// to call again with -1 as base to undo it.
    void shiftIndices(Index base);

// protected:

    /**
    * \brief Get block method
    * @param Line index i and column index j
    * @return Block value if exist or empty Block if not
    **/
    const Block& block(Index i, Index j) const;

    SOFA_ATTRIBUTE_DISABLED__CRS_BLOCK_RENAMING()
    const Block& bloc(Index i, Index j) const
    {
        return block(i, j);
    }

    /**
    * \brief Write block method
    * @param Line index i and column index j
    * @param create, boolean to decide if wblock could add new value into not existing line/column
    * @return pointer on Block value if exist or nullptr if not
    **/
    Block* wblock(Index i, Index j, bool create = false);

    SOFA_ATTRIBUTE_DISABLED__CRS_BLOCK_RENAMING()
    Block* wbloc(Index i, Index j, bool create = false)
    {
        return wblock(i, j, create);
    }

    /**
    * \brief Write block method when rowId and colId are known, this is an optimized wblock specification
    * @param Line index i and column index j
    * @param rowId : Index of value i into rowIndex internal vector
    * @param colId : Index of value j into colIndex internal vector
    * @param create, boolean to decide if wblock could add new value into not existing line/column
    * @return pointer on Block value if exist or nullptr if not
    **/
    Block* wblock(Index i, Index j, Index& rowId, Index& colId, bool create = false);

public:

    const Block& getBlock(Index i, Index j) const;

    const BlockTranspose getSymBlock(Index i, Index j) const;

    void setBlock(Index i, Index j, const Block& v);
    void addBlock(Index i, Index j, const Block& v);

    void setBlock(Index i, Index j, Index& rowId, Index& colId, const Block& v);

    void addBlock(Index i, Index j, Index& rowId, Index& colId, const Block& v);

    Block* getWBlock(Index i, Index j, bool create = false);

    /**
    * \brief Clear row block method. Clear all col of this line.
    * @param i : Line index considering size of matrix in block.
    * \warning if ClearByZeros Policy is activated all col value of line will be set to zero using default constructor
    **/
    void clearRowBlock(Index i);

    /**
    * \brief Clear col block method. Clear this col in all row of matrix.
    * @param j : Col index considering size of matrix in block.
    * \warning if ClearByZeros Policy is activated all col j of each line will be set to zero using default constructor
    **/
    void clearColBlock(Index j);

    std::size_t countEmptyBlocks() const;

    /**
    * \brief Clear both row i and column i in a square matrix
    * @param i : Row and Col index considering size of matrix in block.
    * \warning if ClearByZeros Policy is activated all col i and line i values of will be set to zero using default constructor
    **/
    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void clearRowColBlock(Index i);

    /**
    * \brief Completely clear the matrix
    * \warning if ClearByZeros Policy is activated all value in colsValue will be set to zero using default constructor
    **/
    void clear();


/// @name BlockMatrixWriter operators
/// @{

    void add(unsigned int bi, unsigned int bj, const Block& b);

    void add(unsigned int bi, unsigned int bj, int& rowId, int& colId, const Block& b);

    void addDBlock(unsigned int bi, unsigned int bj, const DBlock& b);

    void addDValue(unsigned int bi, unsigned int bj, const Real b);

    void addDValue(unsigned int bi, unsigned int bj, int& rowId, int& colId, const Real b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiag(unsigned int bi, const Block& b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiag(unsigned int bi, int& rowId, int& colId, const Block &b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiagDBlock(unsigned int bi, const DBlock& b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiagDValue(unsigned int bi, const Real b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiagDValue(unsigned int bi, int& rowId, int& colId, const Real b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSym(unsigned int bi, unsigned int bj, const Block& b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSym(unsigned int bi, unsigned int bj, int& rowId, int& colId, int& rowIdT, int& colIdT, const Block &b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSymDBlock(unsigned int bi, unsigned int bj, const DBlock& b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSymDValue(unsigned int bi, unsigned int bj, const Real b);

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSymDValue(unsigned int bi, unsigned int bj, int& rowId, int& colId, int& rowIdT, int& colIdT, Real b);

/// @}


/// @name Matrix operators
/// @{

    /// Transpose the matrix into res, works only for 3 array variant ("full rows") matrices, ie which can be expressed using the rowBegin, colsIndex and colsValue arrays solely
    template<typename TBlock2, typename TPolicy2>
    void transposeFullRows(CompressedRowSparseMatrixGeneric<TBlock2, TPolicy2>& res) const;

    /** Compute res = this * m
      @warning The block sizes must be compatible, i.e. this::NC==m::NR and res::NR==this::NR and res::NC==m::NC.
      The basic algorithm consists in accumulating rows of m to rows of res: foreach row { foreach col { res[row] += this[row,col] * m[col] } }
      @warning matrices this and m must be compressed
      */
    template<typename RB, typename RP, typename MB, typename MP >
    void mul( CompressedRowSparseMatrixGeneric<RB,RP>& res, const CompressedRowSparseMatrixGeneric<MB,MP>& m ) const;

    static TBlock blockMultTranspose(const TBlock& blockA, const TBlock& blockB);

    /** Compute res = this.transpose * m
      @warning The block sizes must be compatible, i.e. this::NR==m::NR and res::NR==this::NC and res::NC==m::NC
      The basic algorithm consists in accumulating rows of m to rows of res: foreach row { foreach col { res[row] += this[row,col] * m[col] } }
      @warning matrices this and m must be compressed
      */
    template<typename RB, typename RP, typename MB, typename MP >
    void mulTranspose( CompressedRowSparseMatrixGeneric<RB,RP>& res, const CompressedRowSparseMatrixGeneric<MB,MP>& m ) const;

/// @}

    static const char* Name()
    {
        static std::string name = std::string("CompressedRowSparseMatrix") + std::string(traits::Name()); // keep compatibility with previous implementation
        return name.c_str();
    }

    bool check_matrix();

    static bool check_matrix(
        Index nzmax,    // nb values
        Index m,        // number of row
        Index n,        // number of columns
        Index * a_p,    // column pointers (size n+1) or col indices (size nzmax)
        Index * a_i,    // row indices, size nzmax
        Block * a_x      // numerical values, size nzmax
    );

    std::ostream& write(std::ostream& os) const
    {
        os << rowIndex;
        os << rowBegin;
        os << colsIndex;
        os << colsValue;

        return os;
    }

    std::istream& read(std::istream& is)
    {
        {
            std::string line;
            std::getline(is, line);
            std::istringstream(line) >> rowIndex;
        }
        {
            std::string line;
            std::getline(is, line);
            std::istringstream(line) >> rowBegin;
        }
        {
            std::string line;
            std::getline(is, line);
            std::istringstream(line) >> colsIndex;
        }
        {
            std::string line;
            std::getline(is, line);
            std::istringstream(line) >> colsValue;
        }

        return is;
    }

protected:

template<typename TVec>
void writeVector(const TVec& vec, std::ostream& os)
{
    for (auto& v : vec)
        os <<v<<";";
}

template<typename TVec>
void readVector(TVec& vec, std::istream& in)
{
    std::string temp;
    while (std::getline(in, temp, ';'))
    {
        vec.push_back(std::stoi(temp));
    }
}
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIXGENERIC_CPP)
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixGeneric<double>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixGeneric<float>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixGeneric<type::Mat1x1d>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixGeneric<type::Mat1x1f>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixGeneric<type::Mat3x3d>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixGeneric<type::Mat3x3f>;
#endif


} // namespace sofa::linearalgebra
