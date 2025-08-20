#pragma once



void identity()
{
    *this = this->Identity();
}

void clear()
{
    this->setZero();
}

auto ptr()
{
    return this->data();
}

auto ptr() const
{
    return this->data();
}

bool invert(const Matrix& mat)
{
    *this = mat.inverse();
    return true;
}

//auto transposed() const
//{
//    return this->transpose();
//}

auto& x()
{
    if constexpr (IsVectorAtCompileTime && SizeAtCompileTime >= 1)
    {
        return (*this)[0];
    }
    else
    {
        return this->col(0);
    }
}

const auto& x() const
{
    if constexpr (IsVectorAtCompileTime && SizeAtCompileTime >= 1)
    {
        return (*this)[0];
    }
    else
    {
        return this->col(0);
    }
}

auto& y()
{
    if constexpr (IsVectorAtCompileTime && SizeAtCompileTime >= 2)
    {
        return (*this)[1];
    }
    else
    {
        return this->col(1);
    }
}

const auto& y() const
{
    if constexpr (IsVectorAtCompileTime && SizeAtCompileTime >= 2)
    {
        return (*this)[1];
    }
    else
    {
        return this->col(1);
    }
}

auto& z()
{
    if constexpr (IsVectorAtCompileTime && SizeAtCompileTime >= 3)
    {
        return (*this)[2];
    }
    else
    {
        return this->col(2);
    }
}

const auto& z() const
{
    if constexpr (IsVectorAtCompileTime && SizeAtCompileTime >= 3)
    {
        return (*this)[2];
    }
    else
    {
        return this->col(2);
    }
}

//auto operator[](Index i)
//{
//    return (*this)(i);
//}

//const auto& operator[](Index i) const
//{
//    if constexpr (IsVectorAtCompileTime)
//    {
//        return (*this)(0,i);
//    }
//    else
//    {
//        return this->col(i);
//    }
//}

/// Specific set function for 1-element vectors.
void set(const auto r1) noexcept
{
    (*this) << r1;
}

template<typename... ArgsT>
void set(const ArgsT... r) noexcept
{
    (((*this) << r), ...);
}

//template<typename Derived1, typename Derived2>
//auto dot(const Eigen::MatrixBase<Derived1>& a,
//         const Eigen::MatrixBase<Derived2>& b)
//{
//    return a.dot(b);
//}

bool normalizeWithNorm(Matrix::Scalar norm, Matrix::Scalar threshold=std::numeric_limits<Matrix::Scalar>::epsilon())
{
    if (norm>threshold)
    {
        for(auto& s : (*this))
        {
            s /= norm;
        }
        return true;
    }
    else
        return false;
}

/// for square matrices
/// @warning in-place simple symmetrization
/// this = ( this + this.transposed() ) / 2.0
template<int NbLines = Matrix::RowsAtCompileTime, int NbColumns = ColsAtCompileTime>
requires (NbLines == NbColumns)
void symmetrize() noexcept
{
    (*this) = (*this + this->transpose() ) / 2.0;
}

/// l-norm of the vector
/// The type of norm is set by parameter l.
/// Use l<0 for the infinite norm.
auto lNorm( int l ) const
requires (Matrix::IsVectorAtCompileTime == 1)
{
    using Scalar = Matrix::Scalar;

    if( l==2 ) return this->norm(); // euclidean norm
    else if( l<0 ) // infinite norm
    {
        return this->template lpNorm<Eigen::Infinity>();
    }
    else if( l==1 ) // Manhattan norm
    {
        return this->template lpNorm<1>();
    }
    else if( l==0 ) // counting not null
    {
        return static_cast<Scalar>(((*this).array() != 0.0).count());
    }
    else // generic implementation
    {
        //Eigen version l parameter must be known at compile time
        //here, l is given as an argument
        Scalar n = 0;
        for( int i=0; i<SizeAtCompileTime; i++ )
            n += static_cast<Scalar>((pow( std::fabs( (*this)[i] ), l )));
        return static_cast<Scalar>(pow( n, static_cast<Scalar>(1.0)/ static_cast<Scalar>(l) ));
    }
}
