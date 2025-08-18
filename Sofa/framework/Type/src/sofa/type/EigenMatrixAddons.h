#pragma once

struct NoInit{};
inline static constexpr NoInit NOINIT{};

inline static const sofa::Size total_size = Matrix::ColsAtCompileTime * Matrix::RowsAtCompileTime;
inline static const int static_size = Matrix::ColsAtCompileTime * Matrix::RowsAtCompileTime;

using Real = Matrix::Scalar;
static constexpr auto nbLines = Matrix::RowsAtCompileTime;
static constexpr auto nbCols = Matrix::ColsAtCompileTime;
using Size = int;


static constexpr sofa::Size spatial_dimensions = nbLines;
static constexpr sofa::Size coord_total_size = nbLines;
static constexpr sofa::Size deriv_total_size = nbLines*nbCols;

static constexpr auto size()
{
    return static_size;
}

explicit Matrix(const NoInit& noInit)
{}

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
    if constexpr (IsVectorAtCompileTime && static_size >= 1)
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
    if constexpr (IsVectorAtCompileTime && static_size >= 1)
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
    if constexpr (IsVectorAtCompileTime && static_size >= 2)
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
    if constexpr (IsVectorAtCompileTime && static_size >= 2)
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
    if constexpr (IsVectorAtCompileTime && static_size >= 3)
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
    if constexpr (IsVectorAtCompileTime && static_size >= 3)
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

template<typename Derived>
auto linearProduct(const Eigen::MatrixBase<Derived>& vec) const
{
    static_assert(Matrix::IsVectorAtCompileTime && Derived::IsVectorAtCompileTime,
                 "Both arguments must be vectors");
    return (*this).cwiseProduct(vec);
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

template <typename Derived>
auto multTranspose(const Eigen::MatrixBase<Derived>& m) const noexcept
{
    return ((*this).transpose() * m).eval();
}

template <typename Derived>
auto multTransposed(const Eigen::MatrixBase<Derived>& m) const noexcept
{
    return ((*this) * m.transpose()).eval();
}

template <typename Derived>
auto multDiagonal(const Eigen::MatrixBase<Derived>& m) const noexcept
{
    static_assert(Derived::IsVectorAtCompileTime);

    return ((*this) * m.asDiagonal()).eval();
}

template <typename Derived>
void getsub(int L0, int C0, Eigen::MatrixBase<Derived>& m) const noexcept
{
    m = (*this)(seq(L0, Eigen::MatrixBase<Derived>::RowsAtCompileTime), seq(C0, Eigen::MatrixBase<Derived>::ColsAtCompileTime));
}

void getsub(int L0, int C0, Matrix::Scalar& m) const noexcept
{
    m = (*this)(L0,C0);
}

/// for square matrices
/// @warning in-place simple symmetrization
/// this = ( this + this.transposed() ) / 2.0
template<int NbLines = nbLines, int NbColumns = nbCols>
requires (NbLines == NbColumns)
void symmetrize() noexcept
{
    (*this) = (*this + this->transpose() ) / 2.0;
}

// unsafe !
static auto fromPtr(const Real* vptr)
{
    Matrix<Real, nbLines, nbCols> r;
    for(int i=0 ; i < nbCols ; i++)
    {
        r << static_cast<Real>(vptr[i]);
    }
    return r;
}

template<typename Derived>
friend std::istream& operator >> ( std::istream& in, Eigen::MatrixBase<Derived>& matrix);

template<typename Derived>
friend std::ostream& operator << ( std::ostream& out, const Eigen::MatrixBase<Derived>& matrix);

