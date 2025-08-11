#pragma once

struct NoInit;

static constexpr sofa::Size total_size = ColsAtCompileTime;
using Real = Matrix::Scalar;
static constexpr auto nbLines = Matrix::RowsAtCompileTime;
static constexpr auto nbCols = Matrix::ColsAtCompileTime;
using Size = decltype (ColsAtCompileTime);


static constexpr sofa::Size spatial_dimensions = nbCols;
static constexpr sofa::Size coord_total_size = nbLines*nbCols;
static constexpr sofa::Size deriv_total_size = nbLines*nbCols;

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

auto transposed() const
{
    return this->transpose();
}

auto norm2() const
{
    return this->squaredNorm();
}

auto& x()
{
    if constexpr (ColsAtCompileTime == 1 && RowsAtCompileTime >= 1)
    {
        return (*this)(0,0);
    }
    else
    {
        return this->row(0);
    }
}

const auto& x() const
{
    if constexpr (ColsAtCompileTime == 1 && RowsAtCompileTime >= 1)
    {
        return (*this)(0,0);
    }
    else
    {
        return this->row(0);
    }
}

auto& y()
{
    if constexpr (ColsAtCompileTime == 1 && RowsAtCompileTime >= 2)
    {
        return (*this)(0,1);
    }
    else
    {
        return this->row(1);
    }
}

const auto& y() const
{
    if constexpr (ColsAtCompileTime == 1 && RowsAtCompileTime >= 2)
    {
        return (*this)(0,1);
    }
    else
    {
        return this->row(1);
    }
}

auto& z()
{
    if constexpr (ColsAtCompileTime == 1 && RowsAtCompileTime >= 3)
    {
        return (*this)(0,2);
    }
    else
    {
        return this->row(2);
    }
}

const auto& z() const
{
    if constexpr (ColsAtCompileTime == 1 && RowsAtCompileTime >= 3)
    {
        return (*this)(0,2);
    }
    else
    {
        return this->row(2);
    }
}

auto& operator[](Index i)
{
    if constexpr (ColsAtCompileTime == 1)
    {
        return (*this)(i,0);
    }
    else
    {
        return this->col(i);
    }
}

const auto& operator[](Index i) const
{
    if constexpr (ColsAtCompileTime == 1)
    {
        return (*this)(i,0);
    }
    else
    {
        return this->col(i);
    }
}

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

template<typename Derived1, typename Derived2>
auto dot(const Eigen::MatrixBase<Derived1>& a,
         const Eigen::MatrixBase<Derived2>& b)
{
    return a.dot(b);
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

