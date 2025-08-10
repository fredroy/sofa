struct NoInit;

static constexpr sofa::Size total_size = ColsAtCompileTime;

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
        return (*this)(0,i);
    }
    else
    {
        return this->row(i);
    }
}

const auto& operator[](Index i) const
{
    if constexpr (ColsAtCompileTime == 1)
    {
        return (*this)(0,i);
    }
    else
    {
        return this->row(i);
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

template<typename Scalar, int Dim>
auto linearProduct(const Eigen::Ref<const Eigen::Matrix<Scalar, Dim, 1>>& v)
{
    static_assert(Matrix::RowsAtCompileTime == Dim);
    
    decltype(v) r;
    for (std::size_t i=0; i<Dim; i++)
        r[i]=(*this)[i] * v[i];
    return r;
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

