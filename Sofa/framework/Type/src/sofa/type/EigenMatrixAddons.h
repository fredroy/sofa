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

//auto transposed() const
//{
//    return this->transpose();
//}

auto& x()
requires (IsVectorAtCompileTime == 1 && SizeAtCompileTime >= 1)
{
    return (*this)[0];
}

//auto& x()
//requires (IsVectorAtCompileTime == 0 && ColsAtCompileTime >= 1)
//{
//    return (*this).col(0);
//}

const auto& x() const
requires (IsVectorAtCompileTime == 1 && SizeAtCompileTime >= 1)
{
    return (*this)[0];
}

const auto& x() const
requires (IsVectorAtCompileTime == 0 && ColsAtCompileTime >= 1)
{
    return (*this).col(0);
}

auto& y()
requires (IsVectorAtCompileTime == 1 && SizeAtCompileTime >= 2)
{
    return (*this)[1];
}

//auto& y()
//requires (IsVectorAtCompileTime == 0 && ColsAtCompileTime >= 2)
//{
//    return (*this).col(1);
//}

const auto& y() const
requires (IsVectorAtCompileTime == 1 && SizeAtCompileTime >= 2)
{
    return (*this)[1];
}

const auto& y() const
requires (IsVectorAtCompileTime == 0 && ColsAtCompileTime >= 2)
{
    return (*this).col(1);
}

auto& z()
requires (IsVectorAtCompileTime == 1 && SizeAtCompileTime >= 3)
{
    return (*this)[2];
}

//auto& z()
//requires (IsVectorAtCompileTime == 0 && ColsAtCompileTime >= 3)
//{
//    return (*this).col(2);
//}

const auto& z() const
requires (IsVectorAtCompileTime == 1 && SizeAtCompileTime >= 3)
{
    return (*this)[2];
}

const auto& z() const
requires (IsVectorAtCompileTime == 0 && ColsAtCompileTime >= 3)
{
    return (*this).col(2);
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

template<typename OtherDerived, int NbLine = RowsAtCompileTime, int NbColumn = ColsAtCompileTime>
requires (   (OtherDerived::IsVectorAtCompileTime == 1)
          && (OtherDerived::SizeAtCompileTime >= NbColumn-1)
)
static auto transformTranslation(const MatrixBase<OtherDerived>& t) noexcept
{
    static constexpr auto L = RowsAtCompileTime;
    static constexpr auto C = ColsAtCompileTime;

    Matrix<Scalar, L, C> m;
    m.identity();
    for (int i=0; i<C-1; ++i)
        m(i,C-1) = t[i];
    return m;
}

static auto transformScale(Scalar s) noexcept
requires (ColsAtCompileTime == RowsAtCompileTime)
{
    static constexpr auto L = RowsAtCompileTime;
    static constexpr auto C = ColsAtCompileTime;

    Matrix<Scalar, L, C> m;
    m.identity();
    for (int i=0; i<C-1; ++i)
        m(i,i) = s;

    return m;
}

template<typename OtherDerived, int NbColumn = OtherDerived::SizeAtCompileTime>
requires (OtherDerived::IsVectorAtCompileTime == 1)
static auto transformScale(const MatrixBase<OtherDerived>& s) noexcept
{
    using Scalar = typename OtherDerived::Scalar;
    static constexpr auto L = OtherDerived::RowsAtCompileTime;
    static constexpr auto C = OtherDerived::ColsAtCompileTime;

    Matrix<Scalar, L, C> m;
    m.identity();
    for (int i=0; i<C-1; ++i)
        m(i,i) = s[i];
    return m;
}

template<class Quat>
static auto transformRotation(const Quat& q) noexcept
requires( (ColsAtCompileTime == RowsAtCompileTime)
       && ( (ColsAtCompileTime == 3) || (ColsAtCompileTime == 4)) )
{

    static constexpr auto L = RowsAtCompileTime;
    static constexpr auto C = ColsAtCompileTime;

    Matrix<Scalar, L, C> m;
    m.identity();

    if constexpr(L == 4 && C == 4)
    {
        q.toHomogeneousMatrix(m);
        return m;
    }
    else // if constexpr(L == 3 && C == 3)
    {
        q.toMatrix(m);
        return m;
    }
}


/// Inverse Matrix considering the matrix as a transformation.
template<typename OtherDerived>
requires ((Matrix::ColsAtCompileTime == OtherDerived::ColsAtCompileTime)
          && (Matrix::RowsAtCompileTime == OtherDerived::RowsAtCompileTime)
          && (Matrix::RowsAtCompileTime == OtherDerived::RowsAtCompileTime)
          && (std::is_same_v<typename Matrix::Scalar, typename OtherDerived::Scalar>))
bool transformInvert(const Eigen::MatrixBase<OtherDerived>& from)
{
    using Scalar = typename Matrix::Scalar;
    constexpr int Dim = Matrix::RowsAtCompileTime;

    Matrix<Scalar, Dim-1,Dim-1> R, R_inv;
    from.getsub(0,0,R);
    R_inv = R.inverse();

    Matrix<Scalar, Dim-1,1> t, t_inv;
    from.getsub(0,Dim-1,t);
    t_inv = -1.*R_inv*t;

    (*this).setsub(0,0,R_inv);
    (*this).setsub(0,Dim-1,t_inv);
    for (sofa::Size i=0; i<Dim-1; ++i)
        (*this)(Dim-1,i)=0.0;
    (*this)(Dim-1,Dim-1)=1.0;

    return true; // check determinant
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
