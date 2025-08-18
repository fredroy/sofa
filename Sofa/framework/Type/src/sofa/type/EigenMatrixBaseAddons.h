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
