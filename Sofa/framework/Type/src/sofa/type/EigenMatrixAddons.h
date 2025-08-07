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

auto& x()
{
    return this->operator()(0,0);
}

const auto& x() const
{
    return this->operator()(0,0);
}

auto& operator[](Index i)
{
    if constexpr (this->IsRowMajor && ColsAtCompileTime > 1)
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
    if constexpr (this->IsRowMajor && ColsAtCompileTime > 1)
    {
        return this->row(i);
    }
    else
    {
        return this->operator()(i);
    }
}
