// struct NoInit{};
// inline static constexpr NoInit NOINIT{};

using Real = MatrixBase::Scalar;
using Size = int;
using value_type = MatrixBase::Scalar;
using size_type = int;


inline static const int total_size = MatrixBase::ColsAtCompileTime * MatrixBase::RowsAtCompileTime;
inline static const int static_size = (MatrixBase::RowsAtCompileTime == Eigen::Dynamic || MatrixBase::ColsAtCompileTime == Eigen::Dynamic) ? 0 : total_size;
static constexpr int nbLines = MatrixBase::RowsAtCompileTime;
static constexpr int nbCols = MatrixBase::ColsAtCompileTime;
static constexpr int spatial_dimensions = nbLines;
static constexpr int coord_total_size = nbLines;
static constexpr int deriv_total_size = nbLines*nbCols;

static constexpr auto size()
{
    return static_size;
}

explicit MatrixBase(const sofa::type::NoInit& noInit)
: MatrixBase()
{}

void assign(const MatrixBase::Scalar v)
{
    return this->fill(v);
}

auto transposed() const
{
    return this->transpose();
}

auto norm2() const
{
    return this->squaredNorm();
}

auto line(const int i) const
{
    return this->row(i);
}

template <typename OtherDerived>
void getsub(int L0, int C0, MatrixBase<OtherDerived>& m) const noexcept
{
    m = (*this)(seq(L0, OtherDerived::RowsAtCompileTime), seq(C0, OtherDerived::ColsAtCompileTime));
}

template <typename OtherDerived>
void getsub(int L0, MatrixBase<OtherDerived>& m) const noexcept
requires (MatrixBase::IsVectorAtCompileTime == 1 && OtherDerived::IsVectorAtCompileTime == 1 && OtherDerived::SizeAtCompileTime <= MatrixBase::SizeAtCompileTime)
{
    m = (*this)(seq(L0, OtherDerived::ColsAtCompileTime), 0);
}

void getsub(int L0, int C0, MatrixBase::Scalar& m) const noexcept
{
    m = (*this)(L0,C0);
}

void getsub(int L0, MatrixBase::Scalar& m) const noexcept
requires (MatrixBase::IsVectorAtCompileTime == 1)
{
    m = (*this)(L0,0);
}

template <typename OtherDerived>
void setsub(int L0, int C0, const Eigen::MatrixBase<OtherDerived>& m) noexcept
{
    for (int i=0; i<OtherDerived::RowsAtCompileTime; i++)
        for (int j=0; j<OtherDerived::ColsAtCompileTime; j++)
            (*this)(i+L0,j+C0) = m(i,j);
}

template<typename OtherDerived>
auto linearProduct(const Eigen::MatrixBase<OtherDerived>& vec) const
{
    static_assert(MatrixBase::IsVectorAtCompileTime && OtherDerived::IsVectorAtCompileTime,
                 "Both arguments must be vectors");
    return (*this).cwiseProduct(vec);
}

/// Multiplication operator Matrix * Vector considering the matrix as a transformation.
template<typename OtherDerived>
requires (OtherDerived::IsVectorAtCompileTime == 1 && MatrixBase::SizeAtCompileTime >= (OtherDerived::ColsAtCompileTime - 1) * (OtherDerived::ColsAtCompileTime - 1))
auto transform(const Eigen::MatrixBase<OtherDerived>& v) const noexcept
{
    OtherDerived r;
    for(int i=0; i<OtherDerived::ColsAtCompileTime-1; i++)
    {
        r[i]=(*this)(i,0) * v[0];
        for(int j=1; j<OtherDerived::ColsAtCompileTime-1; j++)
            r[i] += (*this)(i,j) * v[j];
        r[i] += (*this)(i,OtherDerived::ColsAtCompileTime-1);
    }
    return r;
}
