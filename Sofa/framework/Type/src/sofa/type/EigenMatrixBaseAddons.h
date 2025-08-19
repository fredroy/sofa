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
void getsub(int L0, int C0, Eigen::MatrixBase<OtherDerived>& m) const noexcept
{
    m = (*this)(seq(L0, OtherDerived::RowsAtCompileTime), seq(C0, OtherDerived::ColsAtCompileTime));
}

void getsub(int L0, int C0, MatrixBase::Scalar& m) const noexcept
{
    m = (*this)(L0,C0);
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
