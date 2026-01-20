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
#include <sofa/linearalgebra/config.h>

#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/BlockFullMatrix.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/matrix_bloc_traits.h>

namespace sofa::linearalgebra
{
    
/// Simple BTD matrix container
template< std::size_t N, typename T>
class BTDMatrix : public linearalgebra::BaseMatrix
{
public:
    enum { BSIZE = N };
    typedef T Real;
    typedef typename linearalgebra::BaseMatrix::Index Index;

    class TransposedBlock
    {
    public:
        const type::Mat<BSIZE,BSIZE,Real>& m;
        TransposedBlock(const type::Mat<BSIZE,BSIZE,Real>& m) : m(m) {}
        type::Vec<BSIZE,Real> operator*(const type::Vec<BSIZE,Real>& v)
        {
            if constexpr (BSIZE == 6)
            {
                // Optimized 6x6 transposed matrix-vector multiplication (m^T * v)
                type::Vec<6,Real> r(type::NOINIT);
                r[0] = m(0,0)*v[0] + m(1,0)*v[1] + m(2,0)*v[2] + m(3,0)*v[3] + m(4,0)*v[4] + m(5,0)*v[5];
                r[1] = m(0,1)*v[0] + m(1,1)*v[1] + m(2,1)*v[2] + m(3,1)*v[3] + m(4,1)*v[4] + m(5,1)*v[5];
                r[2] = m(0,2)*v[0] + m(1,2)*v[1] + m(2,2)*v[2] + m(3,2)*v[3] + m(4,2)*v[4] + m(5,2)*v[5];
                r[3] = m(0,3)*v[0] + m(1,3)*v[1] + m(2,3)*v[2] + m(3,3)*v[3] + m(4,3)*v[4] + m(5,3)*v[5];
                r[4] = m(0,4)*v[0] + m(1,4)*v[1] + m(2,4)*v[2] + m(3,4)*v[3] + m(4,4)*v[4] + m(5,4)*v[5];
                r[5] = m(0,5)*v[0] + m(1,5)*v[1] + m(2,5)*v[2] + m(3,5)*v[3] + m(4,5)*v[4] + m(5,5)*v[5];
                return r;
            }
            else
            {
                return m.multTranspose(v);
            }
        }
        /// Fused operation: M^T * (v1 + v2) - avoids temporary vector creation
        type::Vec<BSIZE,Real> multAdd(const type::Vec<BSIZE,Real>& v1, const type::Vec<BSIZE,Real>& v2) const
        {
            if constexpr (BSIZE == 6)
            {
                type::Vec<6,Real> r(type::NOINIT);
                const Real s0 = v1[0] + v2[0], s1 = v1[1] + v2[1], s2 = v1[2] + v2[2];
                const Real s3 = v1[3] + v2[3], s4 = v1[4] + v2[4], s5 = v1[5] + v2[5];
                r[0] = m(0,0)*s0 + m(1,0)*s1 + m(2,0)*s2 + m(3,0)*s3 + m(4,0)*s4 + m(5,0)*s5;
                r[1] = m(0,1)*s0 + m(1,1)*s1 + m(2,1)*s2 + m(3,1)*s3 + m(4,1)*s4 + m(5,1)*s5;
                r[2] = m(0,2)*s0 + m(1,2)*s1 + m(2,2)*s2 + m(3,2)*s3 + m(4,2)*s4 + m(5,2)*s5;
                r[3] = m(0,3)*s0 + m(1,3)*s1 + m(2,3)*s2 + m(3,3)*s3 + m(4,3)*s4 + m(5,3)*s5;
                r[4] = m(0,4)*s0 + m(1,4)*s1 + m(2,4)*s2 + m(3,4)*s3 + m(4,4)*s4 + m(5,4)*s5;
                r[5] = m(0,5)*s0 + m(1,5)*s1 + m(2,5)*s2 + m(3,5)*s3 + m(4,5)*s4 + m(5,5)*s5;
                return r;
            }
            else
            {
                return m.multTranspose(v1 + v2);
            }
        }
        type::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            type::Mat<BSIZE,BSIZE,Real> r;
            for (Index i=0; i<BSIZE; i++)
                for (Index j=0; j<BSIZE; j++)
                    r(i,j)=-m(j,i);
            return r;
        }
    };

    class Block : public type::Mat<BSIZE,BSIZE,Real>
    {
    public:
        Index Nrows() const { return BSIZE; }
        Index Ncols() const { return BSIZE; }
        void resize(Index, Index)
        {
            this->clear();
        }
        const T& element(Index i, Index j) const { return (*this)(i,j); }
        void set(Index i, Index j, const T& v) { (*this)(i,j) = v; }
        void add(Index i, Index j, const T& v) { (*this)(i,j) += v; }
        void operator=(const type::Mat<BSIZE,BSIZE,Real>& v)
        {
            type::Mat<BSIZE,BSIZE,Real>::operator=(v);
        }
        type::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            type::Mat<BSIZE,BSIZE,Real> r;
            for (Index i=0; i<BSIZE; i++)
                for (Index j=0; j<BSIZE; j++)
                    r(i,j)=-(*this)(i,j);
            return r;
        }
        type::Mat<BSIZE,BSIZE,Real> operator-(const type::Mat<BSIZE,BSIZE,Real>& m) const
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator-(m);
        }
        type::Vec<BSIZE,Real> operator*(const type::Vec<BSIZE,Real>& v)
        {
            if constexpr (BSIZE == 6)
            {
                // Optimized 6x6 matrix-vector multiplication with full unrolling
                type::Vec<6,Real> r(type::NOINIT);
                const auto& m = *this;
                r[0] = m(0,0)*v[0] + m(0,1)*v[1] + m(0,2)*v[2] + m(0,3)*v[3] + m(0,4)*v[4] + m(0,5)*v[5];
                r[1] = m(1,0)*v[0] + m(1,1)*v[1] + m(1,2)*v[2] + m(1,3)*v[3] + m(1,4)*v[4] + m(1,5)*v[5];
                r[2] = m(2,0)*v[0] + m(2,1)*v[1] + m(2,2)*v[2] + m(2,3)*v[3] + m(2,4)*v[4] + m(2,5)*v[5];
                r[3] = m(3,0)*v[0] + m(3,1)*v[1] + m(3,2)*v[2] + m(3,3)*v[3] + m(3,4)*v[4] + m(3,5)*v[5];
                r[4] = m(4,0)*v[0] + m(4,1)*v[1] + m(4,2)*v[2] + m(4,3)*v[3] + m(4,4)*v[4] + m(4,5)*v[5];
                r[5] = m(5,0)*v[0] + m(5,1)*v[1] + m(5,2)*v[2] + m(5,3)*v[3] + m(5,4)*v[4] + m(5,5)*v[5];
                return r;
            }
            else
            {
                return type::Mat<BSIZE,BSIZE,Real>::operator*(v);
            }
        }
        /// Fused operation: M * (v1 + v2) - avoids temporary vector creation
        type::Vec<BSIZE,Real> multAdd(const type::Vec<BSIZE,Real>& v1, const type::Vec<BSIZE,Real>& v2) const
        {
            if constexpr (BSIZE == 6)
            {
                type::Vec<6,Real> r(type::NOINIT);
                const auto& m = *this;
                const Real s0 = v1[0] + v2[0], s1 = v1[1] + v2[1], s2 = v1[2] + v2[2];
                const Real s3 = v1[3] + v2[3], s4 = v1[4] + v2[4], s5 = v1[5] + v2[5];
                r[0] = m(0,0)*s0 + m(0,1)*s1 + m(0,2)*s2 + m(0,3)*s3 + m(0,4)*s4 + m(0,5)*s5;
                r[1] = m(1,0)*s0 + m(1,1)*s1 + m(1,2)*s2 + m(1,3)*s3 + m(1,4)*s4 + m(1,5)*s5;
                r[2] = m(2,0)*s0 + m(2,1)*s1 + m(2,2)*s2 + m(2,3)*s3 + m(2,4)*s4 + m(2,5)*s5;
                r[3] = m(3,0)*s0 + m(3,1)*s1 + m(3,2)*s2 + m(3,3)*s3 + m(3,4)*s4 + m(3,5)*s5;
                r[4] = m(4,0)*s0 + m(4,1)*s1 + m(4,2)*s2 + m(4,3)*s3 + m(4,4)*s4 + m(4,5)*s5;
                r[5] = m(5,0)*s0 + m(5,1)*s1 + m(5,2)*s2 + m(5,3)*s3 + m(5,4)*s4 + m(5,5)*s5;
                return r;
            }
            else
            {
                return type::Mat<BSIZE,BSIZE,Real>::operator*(v1 + v2);
            }
        }
        /// Fused operation: M * (v1 + v2) + v3 - avoids temporary vector creation
        type::Vec<BSIZE,Real> multAddAdd(const type::Vec<BSIZE,Real>& v1, const type::Vec<BSIZE,Real>& v2, const type::Vec<BSIZE,Real>& v3) const
        {
            if constexpr (BSIZE == 6)
            {
                type::Vec<6,Real> r(type::NOINIT);
                const auto& m = *this;
                const Real s0 = v1[0] + v2[0], s1 = v1[1] + v2[1], s2 = v1[2] + v2[2];
                const Real s3 = v1[3] + v2[3], s4 = v1[4] + v2[4], s5 = v1[5] + v2[5];
                r[0] = m(0,0)*s0 + m(0,1)*s1 + m(0,2)*s2 + m(0,3)*s3 + m(0,4)*s4 + m(0,5)*s5 + v3[0];
                r[1] = m(1,0)*s0 + m(1,1)*s1 + m(1,2)*s2 + m(1,3)*s3 + m(1,4)*s4 + m(1,5)*s5 + v3[1];
                r[2] = m(2,0)*s0 + m(2,1)*s1 + m(2,2)*s2 + m(2,3)*s3 + m(2,4)*s4 + m(2,5)*s5 + v3[2];
                r[3] = m(3,0)*s0 + m(3,1)*s1 + m(3,2)*s2 + m(3,3)*s3 + m(3,4)*s4 + m(3,5)*s5 + v3[3];
                r[4] = m(4,0)*s0 + m(4,1)*s1 + m(4,2)*s2 + m(4,3)*s3 + m(4,4)*s4 + m(4,5)*s5 + v3[4];
                r[5] = m(5,0)*s0 + m(5,1)*s1 + m(5,2)*s2 + m(5,3)*s3 + m(5,4)*s4 + m(5,5)*s5 + v3[5];
                return r;
            }
            else
            {
                return type::Mat<BSIZE,BSIZE,Real>::operator*(v1 + v2) + v3;
            }
        }
        type::Mat<BSIZE,BSIZE,Real> operator*(const type::Mat<BSIZE,BSIZE,Real>& m)
        {
            return sofa::type::operator*(*this, m);
        }
        type::Mat<BSIZE,BSIZE,Real> operator*(const TransposedBlock& mt)
        {
            return this->multTransposed(mt.m);
        }
        TransposedBlock t() const
        {
            return TransposedBlock(*this);
        }
        Block i() const
        {
            Block r;
            const bool canInvert = r.invert(*this);
            assert(canInvert);
            SOFA_UNUSED(canInvert);
            return r;
        }
    };

    typedef Block SubMatrixType;
    typedef sofa::type::Mat<N,N,Real> BlockType;
    typedef BlockFullMatrix<N, T> InvMatrixType;

    // return the dimension of submatrices
    constexpr static Index getSubMatrixDim()
    {
        return BSIZE;
    }

protected:
    Block* data;
    Index nTRow,nTCol;
    Index nBRow,nBCol;
    Index allocsize;

public:

    BTDMatrix();

    BTDMatrix(Index nbRow, Index nbCol);

    ~BTDMatrix() override;

    Block* ptr() { return data; }
    const Block* ptr() const { return data; }

    //Real* operator[](Index i)
    //{
    //    return data+i*pitch;
    //}
    const Block& bloc(Index bi, Index bj) const;

    Block& bloc(Index bi, Index bj);

    void resize(Index nbRow, Index nbCol) override;

    Index rowSize(void) const override;

    Index colSize(void) const override;

    SReal element(Index i, Index j) const override;

    const Block& asub(Index bi, Index bj, Index, Index) const;

    const Block& sub(Index i, Index j, Index, Index) const;

    Block& asub(Index bi, Index bj, Index, Index);

    Block& sub(Index i, Index j, Index, Index);

    template<class B>
    void getSubMatrix(Index i, Index j, Index nrow, Index ncol, B& m);

    template<class B>
    void getAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, B& m);

    template<class B>
    void setSubMatrix(Index i, Index j, Index nrow, Index ncol, const B& m);

    template<class B>
    void setAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, const B& m);

    void set(Index i, Index j, double v) override;

    void add(Index i, Index j, double v) override;

    /**
     * Accumulation specialized on contributions of the same size than the blocks.
     */
    template <std::size_t M = N, std::enable_if_t<(M > 3), int> = 0>
    void add(Index row, Index col, const type::Mat<BSIZE, BSIZE, Real>& v);

    void clear(Index i, Index j) override;

    void clearRow(Index i) override;

    void clearCol(Index j) override;

    void clearRowCol(Index i) override;

    void clear() override;

    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res(rowSize());
        for (Index bi=0; bi<nBRow; ++bi)
        {
            const Index b0 = (bi > 0) ? 0 : 1;
            const Index b1 = ((bi < nBRow - 1) ? 3 : 2);
            for (Index i=0; i<BSIZE; ++i)
            {
                Real r = 0;
                for (Index bj = b0; bj < b1; ++bj)
                {
                    for (Index j=0; j<BSIZE; ++j)
                    {
                        r += data(bi*3+bj,i,j) * v[(bi + bj - 1)*BSIZE + j];
                    }
                }
                res[bi*BSIZE + i] = r;
            }
        }
        return res;
    }

    static const char* Name()
    {
        static std::string name { "BTDMatrix" + std::to_string(N) + matrix_bloc_traits<T, Index>::Name() };
        return name.c_str();
    }
};


template <std::size_t N, typename T>
template <std::size_t M, std::enable_if_t<(M > 3), int>>
void BTDMatrix<N, T>::add(Index row, Index col,
    const type::Mat<BSIZE, BSIZE, Real>& v)
{
    if (row % BSIZE == 0 && col % BSIZE == 0)
    {
        const Index bi = row / BSIZE;
        const Index bj = col / BSIZE;
        const Index bindex = bj - bi + 1;
        if (bindex >= 3)
        {
            return;
        }
        data[bi * 3 + bindex] += v;
    }
    else
    {
        for (sofa::Index i = 0; i < BSIZE; ++i)
        {
            for (sofa::Index j = 0; j < BSIZE; ++j)
            {
                this->add(row + i, col + j, v(i, j));
            }
        }
    }
}


#if !defined(SOFA_LINEARALGEBRA_BTDMATRIX_CPP)
extern template class SOFA_LINEARALGEBRA_API linearalgebra::BTDMatrix<6, SReal>;
#endif

} // namespace sofa::linearalgebra
