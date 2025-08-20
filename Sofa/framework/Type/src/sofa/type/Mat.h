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

#include <sofa/type/config.h>
#include <sofa/type/fwd.h>

#include <sofa/type/fixed_array.h>
#include <sofa/type/Vec.h>

#include <iostream>

#define EIGEN_MATRIXBASE_PLUGIN "sofa/type/EigenMatrixBaseAddons.h"
#define EIGEN_MATRIX_PLUGIN "sofa/type/EigenMatrixAddons.h"
#include <Eigen/Dense>


namespace // anonymous
{
    template<typename real>
    real rabs(const real r)
    {
        if constexpr (std::is_signed<real>())
            return std::abs(r);
        else
            return r;
    }

    template<typename real>
    bool equalsZero(const real r, const real epsilon = std::numeric_limits<real>::epsilon())
    {
        return rabs(r) <= epsilon;
    }

} // anonymous namespace

namespace sofa::type
{



template <int L, int C, class real = SReal>
using Mat = Eigen::Matrix<real, L, C, Eigen::AutoAlign | Eigen::ColMajor>;

template <int L, int C, class real = SReal>
using MatNoInit = Mat<L, C, real>;

typedef Mat<1,1,float> Mat1x1f;
typedef Mat<1,1,double> Mat1x1d;

typedef Mat<2,2,float> Mat2x2f;
typedef Mat<2,2,double> Mat2x2d;

typedef Mat<3,3,float> Mat3x3f;
typedef Mat<3,3,double> Mat3x3d;

typedef Mat<3,4,float> Mat3x4f;
typedef Mat<3,4,double> Mat3x4d;

typedef Mat<4,4,float> Mat4x4f;
typedef Mat<4,4,double> Mat4x4d;

typedef Mat<6, 6, float> Mat6x6f;
typedef Mat<6, 6, double> Mat6x6d;

typedef Mat<2,2,SReal> Mat2x2;
typedef Mat<3,3,SReal> Mat3x3;
typedef Mat<4,4,SReal> Mat4x4;
typedef Mat<6,6,SReal> Mat6x6;

typedef Mat<2,2,SReal> Matrix2;
typedef Mat<3,3,SReal> Matrix3;
typedef Mat<4,4,SReal> Matrix4;

template<typename Derived>
auto oneNorm(const Eigen::MatrixBase<Derived>& m)
{
    return m.cwiseAbs().colwise().sum().maxCoeff();
}

template<typename Derived>
auto infNorm(const Eigen::MatrixBase<Derived>& m)
{
    return m.cwiseAbs().rowwise().sum().maxCoeff();
}

template<typename Derived>
auto determinant(const Eigen::MatrixBase<Derived>& m)
{
    return m.determinant();
}

template<typename Derived>
auto trace(const Eigen::MatrixBase<Derived>& m)
{
    return m.trace();
}

template<typename Derived1, typename Derived2>
bool invertMatrix(Eigen::MatrixBase<Derived1>& dest, const Eigen::MatrixBase<Derived2>& from)
{
    const bool isInvertible = true;// check determinant
    dest = from.inverse();

    return isInvertible;
}

/// Create a matrix as \f$ u v^T \f$
template<typename DerivedU, typename DerivedV>
auto dyad(const Eigen::MatrixBase<DerivedU>& u,
          const Eigen::MatrixBase<DerivedV>& v) {
    return u * v.transpose();
}

/// generic function (used for Rigid* types)
template <class Tu, class Tv, int N = Tu::size(), int M = Tv::size()>
requires (std::is_same_v<typename Tu::value_type, typename Tv::value_type>)
auto dyad(const Tu& u, const Tv& v) noexcept
{
    Mat<N, M, typename Tu::value_type> res;
    for (sofa::Size i = 0; i < Tu::size(); ++i)
    {
        for (sofa::Size j = 0; j < Tv::size(); ++j)
        {
            res(i,j) = u[i] * v[j];
        }
    }
    return res;
}


template<typename Real, typename Derived>
requires (Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == 3)
auto crossProductMatrix(const Eigen::MatrixBase<Derived>& v) noexcept
{
    type::Mat<3, 3, Real> res;
    res(0,0)=0;
    res(0,1)=-v[2];
    res(0,2)=v[1];
    res(1,0)=v[2];
    res(1,1)=0;
    res(1,2)=-v[0];
    res(2,0)=-v[1];
    res(2,1)=v[0];
    res(2,2)=0;
    return res;
}

template<typename Derived>
requires (Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == 3)
auto crossProductMatrix(const Eigen::MatrixBase<Derived>& v) noexcept
{
    return crossProductMatrix<typename Eigen::MatrixBase<Derived>::Scalar>(v);
}

/// Inverse Matrix considering the matrix as a transformation.
template<typename Derived1, typename Derived2>
requires ((Derived1::ColsAtCompileTime == Derived2::ColsAtCompileTime)
          && (Derived1::RowsAtCompileTime == Derived2::RowsAtCompileTime)
          && (Derived1::RowsAtCompileTime == Derived1::RowsAtCompileTime)
          && (std::is_same_v<typename Derived1::Scalar, typename Derived2::Scalar>))
bool transformInvertMatrix(Eigen::MatrixBase<Derived1>& dest, const Eigen::MatrixBase<Derived2>& from)
{
    return dest.transformInvert(from);
}

/// return a * b^T
template<typename Derived1, typename Derived2>
requires ( (Derived1::IsVectorAtCompileTime == 1) && (Derived2::IsVectorAtCompileTime == 1)
        && (Derived1::SizeAtCompileTime == Derived2::SizeAtCompileTime))
auto tensorProduct(const Eigen::MatrixBase<Derived1>& a, const Eigen::MatrixBase<Derived2>& b ) noexcept
{
    return a * b.transpose();
}

} // namespace sofa::type

namespace std
{

template<typename Derived>
requires (Derived::IsVectorAtCompileTime == 0)
std::istream& operator >> ( std::istream& in, Eigen::MatrixBase<Derived>& m )
{
    auto c = in.peek();
    while (c==' ' || c=='\n' || c=='[')
    {
      in.get();
      if( c=='[' ) break;
      c = in.peek();
    }

    // first column
    for (sofa::Size i=0; i<Derived::RowsAtCompileTime; i++)
    {
        in >> m(i, 0);
    }

    for (sofa::Size i=1; i<Derived::ColsAtCompileTime; i++)
    {
      c = in.peek();
      while (c==' ' || c==',')
      {
          in.get();
          c = in.peek();
      }
      for (sofa::Size j=0; j<Derived::RowsAtCompileTime; j++)
      {
          in >> m(i, j);
      }
    }
    if(in.eof()) return in;
    c = in.peek();
    while (c==' ' || c=='\n' || c==']')
    {
      in.get();
      if( c==']' ) break;
      if(in.eof()) break;
      c = in.peek();
    }
    return in;
}

template<typename Derived>
requires (Derived::IsVectorAtCompileTime == 0)
std::ostream& operator << ( std::ostream& out, const Eigen::MatrixBase<Derived>& m )
{
    out << '[' << m(0);
    for (int i=1; i<Derived::ColsAtCompileTime; i++)
      out << ',' << m.col(i);
    out << ']';
    return out;
}

} // namespace std

