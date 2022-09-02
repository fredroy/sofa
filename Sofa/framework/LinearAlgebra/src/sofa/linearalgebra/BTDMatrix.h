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
#include <sofa/linearalgebra/BlocFullMatrix.h>
#include <sofa/linearalgebra/FullVector.h>

namespace sofa::linearalgebra
{
    
/// Simple BTD matrix container
template< std::size_t N, typename T>
class BTDMatrix : public linearalgebra::BaseMatrix
{
public:
    static constexpr sofa::Size BSIZE = N;

    typedef T Real;
    typedef typename linearalgebra::BaseMatrix::Index Index;

    class TransposedBlock
    {
    public:
        const type::Mat<BSIZE,BSIZE,Real>& m;
        constexpr TransposedBlock(const type::Mat<BSIZE,BSIZE,Real>& m) : m(m) {}
        constexpr type::Vec<BSIZE,Real> operator*(const type::Vec<BSIZE,Real>& v)
        {
            return m.multTranspose(v);
        }
        constexpr type::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            type::Mat<BSIZE,BSIZE,Real> r;
            for (Index i=0; i<BSIZE; i++)
                for (Index j=0; j<BSIZE; j++)
                    r[i][j]=-m[j][i];
            return r;
        }
    };

    class Block : public type::Mat<BSIZE,BSIZE,Real>
    {
    public:
        constexpr Index Nrows() const { return BSIZE; }
        constexpr Index Ncols() const { return BSIZE; }
        constexpr void resize(Index, Index)
        {
            this->clear();
        }
        constexpr const T& element(Index i, Index j) const { return (*this)[i][j]; }
        constexpr void set(Index i, Index j, const T& v) { (*this)[i][j] = v; }
        constexpr void add(Index i, Index j, const T& v) { (*this)[i][j] += v; }
        constexpr void operator=(const type::Mat<BSIZE,BSIZE,Real>& v)
        {
            type::Mat<BSIZE,BSIZE,Real>::operator=(v);
        }
        constexpr type::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            type::Mat<BSIZE,BSIZE,Real> r;
            for (Index i=0; i<BSIZE; i++)
                for (Index j=0; j<BSIZE; j++)
                    r[i][j]=-(*this)[i][j];
            return r;
        }
        constexpr type::Mat<BSIZE,BSIZE,Real> operator-(const type::Mat<BSIZE,BSIZE,Real>& m) const
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator-(m);
        }
        
        constexpr type::Vec<BSIZE,Real> operator*(const type::Vec<BSIZE,Real>& v)
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator*(v);
        }
        
        constexpr type::Mat<BSIZE,BSIZE,Real> operator*(const type::Mat<BSIZE,BSIZE,Real>& m)
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator*(m);
        }
        
        constexpr type::Mat<BSIZE,BSIZE,Real> operator*(const Block& m)
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator*(m);
        }
        
        constexpr type::Mat<BSIZE,BSIZE,Real> operator*(const TransposedBlock& mt)
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator*(mt.m.transposed());
        }
        
        constexpr TransposedBlock t() const
        {
            return TransposedBlock(*this);
        }
        
        constexpr Block i() const
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
    typedef BlocFullMatrix<N, T> InvMatrixType;
    // return the dimension of submatrices when requesting a given size
    static constexpr Index getSubMatrixDim(Index) { return BSIZE; }

protected:
    Block* data;
    Index nTRow,nTCol;
    Index nBRow,nBCol;
    Index allocsize;

public:

    constexpr BTDMatrix()
        : data(nullptr), nTRow(0), nTCol(0), nBRow(0), nBCol(0), allocsize(0)
    {
    }

    constexpr BTDMatrix(Index nbRow, Index nbCol)
        : data(new T[3 * (nbRow / BSIZE)]), nTRow(nbRow), nTCol(nbCol), nBRow(nbRow / BSIZE), nBCol(nbCol / BSIZE), allocsize(3 * (nbRow / BSIZE))
    {
    }

    ~BTDMatrix() override
    {
        if (allocsize > 0)
            delete[] data;
    }

    constexpr Block* ptr() { return data; }
    constexpr const Block* ptr() const { return data; }

    constexpr  const Block& bloc(Index bi, Index bj) const
    {
        return data[3 * bi + (bj - bi + 1)];
    }

    constexpr Block& bloc(Index bi, Index bj)
    {
        return data[3 * bi + (bj - bi + 1)];
    }

    void resize(Index nbRow, Index nbCol) override
    {
        if (nbCol != nTCol || nbRow != nTRow)
        {
            if (allocsize < 0)
            {
                if ((nbRow / BSIZE) * 3 > -allocsize)
                {
                    msg_error("BTDLinearSolver") << "Cannot resize preallocated matrix to size (" << nbRow << "," << nbCol << ")";
                    return;
                }
            }
            else
            {
                if ((nbRow / BSIZE) * 3 > allocsize)
                {
                    if (allocsize > 0)
                        delete[] data;
                    allocsize = (nbRow / BSIZE) * 3;
                    data = new Block[allocsize];
                }
            }
            nTCol = nbCol;
            nTRow = nbRow;
            nBCol = nbCol / BSIZE;
            nBRow = nbRow / BSIZE;
        }
        clear();
    }

    Index rowSize(void) const override
    {
        return nTRow;
    }

    Index colSize(void) const override
    {
        return nTCol;
    }

    SReal element(Index i, Index j) const override
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return (SReal)0;
        return data[bi * 3 + bindex][i][j];
    }

    constexpr const Block& asub(Index bi, Index bj, Index, Index) const
    {
        const Index bindex = bj - bi + 1;
        return data[bi * 3 + bindex];
    }

    constexpr const Block& sub(Index i, Index j, Index, Index) const
    {
        return asub(i / BSIZE, j / BSIZE);
    }

    constexpr Block& asub(Index bi, Index bj, Index, Index)
    {
        const Index bindex = bj - bi + 1;
        return data[bi * 3 + bindex];
    }

    constexpr Block& sub(Index i, Index j, Index, Index)
    {
        return asub(i / BSIZE, j / BSIZE);
    }

    template<class B>
    constexpr void getSubMatrix(Index i, Index j, Index nrow, Index ncol, B& m)
    {
        m = sub(i, j, nrow, ncol);
    }

    template<class B>
    constexpr void getAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, B& m)
    {
        m = asub(bi, bj, nrow, ncol);
    }

    template<class B>
    constexpr void setSubMatrix(Index i, Index j, Index nrow, Index ncol, const B& m)
    {
        sub(i, j, nrow, ncol) = m;
    }

    template<class B>
    constexpr void setAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, const B& m)
    {
        asub(bi, bj, nrow, ncol) = m;
    }

    void set(Index i, Index j, double v) override
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return;
        data[bi * 3 + bindex][i][j] = (Real)v;
    }

    void add(Index i, Index j, double v) override
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return;
        data[bi * 3 + bindex][i][j] += (Real)v;
    }

    void clear(Index i, Index j) override
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return;
        data[bi * 3 + bindex][i][j] = (Real)0;
    }

    void clearRow(Index i) override
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        for (Index bj = 0; bj < 3; ++bj)
            for (Index j = 0; j < BSIZE; ++j)
                data[bi * 3 + bj][i][j] = (Real)0;
    }

    void clearCol(Index j) override
    {
        Index bj = j / BSIZE; j = j % BSIZE;
        if (bj > 0)
            for (Index i = 0; i < BSIZE; ++i)
                data[(bj - 1) * 3 + 2][i][j] = (Real)0;
        for (Index i = 0; i < BSIZE; ++i)
            data[bj * 3 + 1][i][j] = (Real)0;
        if (bj < nBRow - 1)
            for (Index i = 0; i < BSIZE; ++i)
                data[(bj + 1) * 3 + 0][i][j] = (Real)0;
    }

    void clearRowCol(Index i) override
    {
        clearRow(i);
        clearCol(i);
    }

    void clear() override
    {
        for (Index i = 0; i < 3 * nBRow; ++i)
            data[i].clear();
    }

    template<class Real2>
    constexpr FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res(nTRow);
        for (Index bi=0; bi<nBRow; ++bi)
        {
            Index b0 = (bi > 0) ? 0 : 1;
            Index b1 = ((bi < nBRow - 1) ? 3 : 2);
            for (Index i=0; i<BSIZE; ++i)
            {
                Real r = 0;
                for (Index bj = b0; bj < b1; ++bj)
                {
                    for (Index j=0; j<BSIZE; ++j)
                    {
                        r += data[bi*3+bj][i][j] * v[(bi + bj - 1)*BSIZE + j];
                    }
                }
                res[bi*BSIZE + i] = r;
            }
        }
        return res;
    }

    static const char* Name();
};

} // namespace sofa::linearalgebra
