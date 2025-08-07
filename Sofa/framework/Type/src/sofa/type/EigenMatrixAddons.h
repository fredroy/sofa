struct NoInit;

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
    if constexpr (Matrix::IsRowMajor && ColsAtCompileTime >= 1)
    {
        return this->row(0);
    }
    else
    {
        return (*this)(0,0);
    }
}

const auto& x() const
{
    if constexpr (Matrix::IsRowMajor && ColsAtCompileTime >= 1)
    {
        return this->row(0);
    }
    else
    {
        return (*this)(0,0);
    }
}

auto& y()
{
    if constexpr (Matrix::IsRowMajor && ColsAtCompileTime >= 2)
    {
        return this->row(1);
    }
    else
    {
        return (*this)(1,0);
    }
}

const auto& y() const
{
    if constexpr (Matrix::IsRowMajor && ColsAtCompileTime >= 2)
    {
        return this->row(1);
    }
    else
    {
        return (*this)(1,0);
    }
}

auto& z()
{
    if constexpr (Matrix::IsRowMajor && ColsAtCompileTime >= 3)
    {
        return this->row(2);
    }
    else
    {
        return (*this)(2,0);
    }
}

const auto& z() const
{
    if constexpr (Matrix::IsRowMajor && ColsAtCompileTime >= 3)
    {
        return this->row(2);
    }
    else
    {
        return (*this)(2,0);
    }
}

auto& operator[](Index i)
{
    if constexpr (Matrix::IsRowMajor && ColsAtCompileTime > 1)
    {
        return this->row(i);
    }
    else
    {
        return this->operator()(i);
    }
}

const auto& operator[](Index i) const
{
    if constexpr (Matrix::IsRowMajor && ColsAtCompileTime > 1)
    {
        return this->row(i);
    }
    else
    {
        return this->operator()(i);
    }
}
