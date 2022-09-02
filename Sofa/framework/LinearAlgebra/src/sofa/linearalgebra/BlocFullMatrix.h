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
#include <sofa/linearalgebra/FullVector.h>

namespace sofa::linearalgebra
{

/// Simple block full matrix container (used for InvMatrixType)
template< std::size_t N, typename T>
class BlocFullMatrix : public linearalgebra::BaseMatrix
{
public:
    static constexpr sofa::Size BSIZE = N;

    typedef T Real;

    class TransposedBlock
    {

    public:
        const type::Mat<BSIZE,BSIZE,Real>& m;

        TransposedBlock(const sofa::type::Mat<BSIZE, BSIZE, Real>& m_a) : m(m_a) {}

        constexpr type::Vec<BSIZE,Real> operator*(const type::Vec<BSIZE,Real>& v)
        {
            return m.multTranspose(v);
        }

        constexpr type::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            return -m.transposed();
        }
    };

    class Block : public type::Mat<BSIZE,BSIZE,Real>
    {
    public:
        constexpr Index Nrows() const
        {
            return BSIZE;
        }

        constexpr Index Ncols() const
        {
            return BSIZE;
        }

        constexpr void resize(Index, Index)
        {
            clear();
        }

        constexpr const T& element(Index i, Index j) const
        {
            return (*this)[i][j];
        }

        constexpr void set(Index i, Index j, const T& v)
        {
            (*this)[i][j] = v;
        }

        constexpr void add(Index i, Index j, const T& v)
        {
            (*this)[i][j] += v;
        }

        constexpr void operator=(const type::Mat<BSIZE,BSIZE,Real>& v)
        {
            type::Mat<BSIZE,BSIZE,Real>::operator=(v);
        }
        constexpr type::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator-();
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
            r.invert(*this);
            return r;
        }
    };
    typedef Block SubMatrixType;
    // return the dimension of submatrices when requesting a given size
    static constexpr Index getSubMatrixDim(Index)
    {
        return BSIZE;
    }

protected:
    Block* data;
    Index nTRow,nTCol;
    Index nBRow,nBCol;
    Index allocsize;

public:

    constexpr BlocFullMatrix()
        : data(nullptr), nTRow(0), nTCol(0), nBRow(0), nBCol(0), allocsize(0)
    {
    }

    constexpr BlocFullMatrix(Index nbRow, Index nbCol)
        : data(new T[nbRow * nbCol]), nTRow(nbRow), nTCol(nbCol), nBRow(nbRow / BSIZE), nBCol(nbCol / BSIZE), allocsize((nbCol / BSIZE)* (nbRow / BSIZE))
    {
    }

    ~BlocFullMatrix() override
    {
        if (allocsize > 0)
            delete[] data;
    }

    constexpr Block* ptr() { return data; }
    constexpr const Block* ptr() const { return data; }

    constexpr const Block& bloc(Index bi, Index bj) const
    {
        return data[bi * nBCol + bj];
    }

    constexpr Block& bloc(Index bi, Index bj)
    {
        return data[bi * nBCol + bj];
    }

    void resize(Index nbRow, Index nbCol) override
    {
        if (nbCol != nTCol || nbRow != nTRow)
        {
            if (allocsize < 0)
            {
                if ((nbCol / BSIZE) * (nbRow / BSIZE) > -allocsize)
                {
                    msg_error("BTDLinearSolver") << "Cannot resize preallocated matrix to size (" << nbRow << "," << nbCol << ").";
                    return;
                }
            }
            else
            {
                if ((nbCol / BSIZE) * (nbRow / BSIZE) > allocsize)
                {
                    if (allocsize > 0)
                        delete[] data;
                    allocsize = (nbCol / BSIZE) * (nbRow / BSIZE);
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
        return bloc(bi, bj)[i][j];
    }

    constexpr const Block& asub(Index bi, Index bj, Index, Index) const
    {
        return bloc(bi, bj);
    }

    constexpr const Block& sub(Index i, Index j, Index, Index) const
    {
        return asub(i / BSIZE, j / BSIZE);
    }

    constexpr Block& asub(Index bi, Index bj, Index, Index)
    {
        return bloc(bi, bj);
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
        bloc(bi, bj)[i][j] = (Real)v;
    }

    void add(Index i, Index j, double v) override
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        bloc(bi, bj)[i][j] += (Real)v;
    }

    void clear(Index i, Index j) override
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        bloc(bi, bj)[i][j] = (Real)0;
    }

    void clearRow(Index i) override
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        for (Index bj = 0; bj < nBCol; ++bj)
            for (Index j = 0; j < BSIZE; ++j)
                bloc(bi, bj)[i][j] = (Real)0;
    }

    void clearCol(Index j) override
    {
        Index bj = j / BSIZE; j = j % BSIZE;
        for (Index bi = 0; bi < nBRow; ++bi)
            for (Index i = 0; i < BSIZE; ++i)
                bloc(bi, bj)[i][j] = (Real)0;
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
            Index bj = 0;
            for (Index i=0; i<BSIZE; ++i)
            {
                Real r = 0;
                for (Index j=0; j<BSIZE; ++j)
                {
                    r += bloc(bi,bj)[i][j] * v[(bi + bj - 1)*BSIZE + j];
                }
                res[bi*BSIZE + i] = r;
            }
            for (++bj; bj<nBCol; ++bj)
            {
                for (Index i=0; i<BSIZE; ++i)
                {
                    Real r = 0;
                    for (Index j=0; j<BSIZE; ++j)
                    {
                        r += bloc(bi,bj)[i][j] * v[(bi + bj - 1)*BSIZE + j];
                    }
                    res[bi*BSIZE + i] += r;
                }
            }
        }
        return res;
    }


    static constexpr const char* Name();
};

} // namespace sofa::linearalgebra
