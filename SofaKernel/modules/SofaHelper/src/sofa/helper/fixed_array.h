/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/* The following code declares class array,
 * an STL container (as wrapper) for arrays of constant size.
 *
 * See
 *      http://www.josuttis.com/cppcode
 * for details and the latest version.
 *
 * (C) Copyright Nicolai M. Josuttis 2001.
 * Permission to copy, use, modify, sell and distribute this software
 * is granted provided this copyright notice appears in all copies.
 * This software is provided "as is" without express or implied
 * warranty, and with no claim as to its suitability for any purpose.
 *
 * 16 Mar 2017 - stop printing an extra space at end of <<.
 * 17 Jan 2017 - add std::enable_if to replace static_assert (Damien Marchal)
 * 29 Jun 2005 - remove boost includes and reverse iterators. (Jeremie Allard)
 * 23 Aug 2002 - fix for Non-MSVC compilers combined with MSVC libraries.
 * 05 Aug 2001 - minor update (Nico Josuttis)
 * 20 Jan 2001 - STLport fix (Beman Dawes)
 * 29 Sep 2000 - Initial Revision (Nico Josuttis)
 */

// See http://www.boost.org/libs/array for Documentation.

// FF added operator <
// JA added constructors from tuples
#ifndef SOFA_HELPER_FIXED_ARRAY_H
#define SOFA_HELPER_FIXED_ARRAY_H

#include <sofa/helper/config.h>

#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <array>

namespace sofa
{

namespace helper
{

template<class T, std::size_t N>
class fixed_array : public std::array<T, N>
{
public:
    using Inherit = std::array<T, N>;

    typedef typename Inherit::value_type value_type;
    typedef typename Inherit::size_type size_type;
    typedef typename Inherit::reference reference;
    typedef typename Inherit::const_reference const_reference;
    typedef typename Inherit::iterator iterator;
    typedef typename Inherit::const_iterator const_iterator;

public:
    //fixed_array()
    //{
    //}

    ///// Specific constructor for 1-element vectors.
    //template<int NN = N, typename std::enable_if<NN==1,int>::type = 0>
    //explicit fixed_array(value_type r1)
    //{
    //    static_assert(N == 1, "");
    //    (*this)[0] = r1;
    //}

    ///// Specific constructor for 2-elements vectors.
    //template<int NN = N, typename std::enable_if<NN==2,int>::type = 0>
    //fixed_array(value_type r1, value_type r2)
    //{
    //    static_assert(N == 2, "");
    //    (*this)[0] = r1;
    //    (*this)[1] = r2;
    //}

    ///// Specific constructor for 3-elements vectors.
    //template<int NN = N, typename std::enable_if<NN==3,int>::type = 0>
    //fixed_array(value_type r1, value_type r2, value_type r3)
    //{
    //    static_assert(N == 3, "");
    //    (*this)[0] = r1;
    //    (*this)[1] = r2;
    //    (*this)[2] = r3;
    //}

    ///// Specific constructor for 4-elements vectors.
    //template<int NN = N, typename std::enable_if<NN==4,int>::type = 0>
    //fixed_array(value_type r1, value_type r2, value_type r3, value_type r4)
    //{
    //    static_assert(N == 4, "");
    //    (*this)[0] = r1;
    //    (*this)[1] = r2;
    //    (*this)[2] = r3;
    //    (*this)[3] = r4;
    //}

    ///// Specific constructor for 5-elements vectors.
    //template<int NN = N, typename std::enable_if<NN==5,int>::type = 0>
    //fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5)
    //{
    //    static_assert(N == 5, "");
    //    (*this)[0] = r1;
    //    (*this)[1] = r2;
    //    (*this)[2] = r3;
    //    (*this)[3] = r4;
    //    (*this)[4] = r5;
    //}

    ///// Specific constructor for 6-elements vectors.
    //template<int NN = N, typename std::enable_if<NN==6,int>::type = 0>
    //fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6)
    //{
    //    static_assert(N == 6, "");
    //    (*this)[0] = r1;
    //    (*this)[1] = r2;
    //    (*this)[2] = r3;
    //    (*this)[3] = r4;
    //    (*this)[4] = r5;
    //    (*this)[5] = r6;
    //}

    ///// Specific constructor for 7-elements vectors.
    //template<int NN = N, typename std::enable_if<NN==7,int>::type = 0>
    //fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7)
    //{
    //    static_assert(N == 7, "");
    //    (*this)[0] = r1;
    //    (*this)[1] = r2;
    //    (*this)[2] = r3;
    //    (*this)[3] = r4;
    //    (*this)[4] = r5;
    //    (*this)[5] = r6;
    //    (*this)[6] = r7;
    //}

    ///// Specific constructor for 8-elements vectors.
    //template<int NN = N, typename std::enable_if<NN==8,int>::type = 0>
    //fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7, value_type r8)
    //{
    //    static_assert(N == 8, "");
    //    (*this)[0] = r1;
    //    (*this)[1] = r2;
    //    (*this)[2] = r3;
    //    (*this)[3] = r4;
    //    (*this)[4] = r5;
    //    (*this)[5] = r6;
    //    (*this)[6] = r7;
    //    (*this)[7] = r8;
    //}

    ///// Specific constructor for 9-elements vectors.
    //template<int NN = N, typename std::enable_if<NN==9,int>::type = 0>
    //fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7, value_type r8, value_type r9)
    //{
    //    static_assert(N == 9, "");
    //    (*this)[0] = r1;
    //    (*this)[1] = r2;
    //    (*this)[2] = r3;
    //    (*this)[3] = r4;
    //    (*this)[4] = r5;
    //    (*this)[5] = r6;
    //    (*this)[6] = r7;
    //    (*this)[7] = r8;
    //    (*this)[8] = r9;
    //}

    ///// Specific constructor for 10-elements vectors.
    //template<int NN = N, typename std::enable_if<NN==10,int>::type = 0>
    //fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7, value_type r8, value_type r9, value_type r10)
    //{
    //    static_assert(N == 10, "");
    //    (*this)[0] = r1;
    //    (*this)[1] = r2;
    //    (*this)[2] = r3;
    //    (*this)[3] = r4;
    //    (*this)[4] = r5;
    //    (*this)[5] = r6;
    //    (*this)[6] = r7;
    //    (*this)[7] = r8;
    //    (*this)[8] = r9;
    //    (*this)[8] = r10;
    //}

//
//    // iterator support
//    iterator begin()
//    {
//        return Inherit::begin();
//    }
//    const_iterator begin() const
//    {
//        return Inherit::cbegin();
//    }
//    iterator end()
//    {
//        return Inherit::end();
//    }
//    const_iterator end() const
//    {
//        return Inherit::cend();
//    }
//
    // operator[]
//    reference operator[](size_type i)
//    {
//#ifndef NDEBUG
//        assert(i<N && "index in fixed_array must be smaller than size");
//#endif
//        return Inherit::operator[](i);
//    }
//    const_reference operator[](size_type i) const
//    {
//#ifndef NDEBUG
//        assert(i<N && "index in fixed_array must be smaller than size");
//#endif
//        return Inherit::operator[](i);
//    }
//
//    // at() with range check
//    reference at(size_type i)
//    {
//        return Inherit::at(i);
//    }
//    const_reference at(size_type i) const
//    {
//        return Inherit::at(i);
//    }
//
//    // front() and back()
//    reference front()
//    {
//        return Inherit::front();
//    }
//    const_reference front() const
//    {
//        return Inherit::front();
//    }
//    reference back()
//    {
//        return Inherit::back();
//    }
//    const_reference back() const
//    {
//        return Inherit::back();
//    }
//
//    // size is constant
//    constexpr size_type size() const noexcept
//    {
//        return Inherit::size();
//    }
//
//    constexpr bool empty() const noexcept
//    {
//        return Inherit::empty();
//    }
//
//    size_type max_size()
//    {
//        return Inherit::max_size();
//    }
//
//    // swap (note: linear complexity)
//    void swap (fixed_array<T,N>& y)
//    {
//        Inherit::swap(y);
//    }
//
//    // direct access to data
//    constexpr T* data() noexcept
//    {
//        return Inherit::data();
//    }
//
//    // direct access to data
//    constexpr const T* data() const noexcept
//    {
//        return Inherit::data();
//    }
//
//    // assignment with type conversion
//    template <typename T2>
//    fixed_array<T,N>& operator= (const fixed_array<T2,N>& rhs)
//    {
//        //std::copy(rhs.begin(),rhs.end(), begin());
//        for (size_type i=0; i<N; i++)
//            (*this)[i] = rhs[i];
//        return *this;
//    }
//
//    // assign one value to all elements
//    inline void assign (const T& value)
//    {
//        //std::fill_n(begin(),size(),value);
//        for (size_type i=0; i<N; i++)
//            (*this)[i] = value;
//    }

    //inline friend std::ostream& operator << (std::ostream& out, const fixed_array<T,N>& a)
    //{
    //    static_assert(N>0, "Cannot create a zero size arrays") ;
    //    for( size_type i=0; i<N-1; i++ )
    //        out << a[i]<<" ";
    //    out << a[N-1];
    //    return out;
    //}

    //inline friend std::istream& operator >> (std::istream& in, fixed_array<T,N>& a)
    //{
    //    for( size_type i=0; i<N; i++ )
    //        in>>a[i];
    //    return in;
    //}

    //inline bool operator < (const fixed_array& v ) const
    //{
    //    for( size_type i=0; i<N; i++ )
    //    {
    //        if((*this)[i]<v[i] )
    //            return true;  // (*this)<v
    //        else if((*this)[i]>v[i] )
    //            return false; // (*this)>v
    //    }
    //    return false; // (*this)==v
    //}

    // not defined in std::array
    //static const size_type static_size = N;
    //static constexpr size_type static_size() { return N; };

private:

};
//
//template <typename T, size_t N>
//inline std::ostream& operator << (std::ostream& out, const fixed_array<T,N>& a)
//{
//    static_assert(N>0, "Cannot create a zero size arrays") ;
//    for( auto i=0; i<N-1; i++ )
//        out << a[i]<<" ";
//    out << a[N-1];
//    return out;
//}
//
//template <typename T, size_t N>
//inline std::istream& operator >> (std::istream& in, fixed_array<T,N>& a)
//{
//    for( auto i=0; i<N; i++ )
//        in>>a[i];
//    return in;
//}
//
//template<class T>
//inline fixed_array<T, 2> make_array(const T& v0, const T& v1)
//{
//    fixed_array<T, 2> v;
//    v[0] = v0;
//    v[1] = v1;
//    return v;
//}
//
//template<class T>
//inline fixed_array<T, 3> make_array(const T& v0, const T& v1, const T& v2)
//{
//    fixed_array<T, 3> v;
//    v[0] = v0;
//    v[1] = v1;
//    v[2] = v2;
//    return v;
//}
//
//template<class T>
//inline fixed_array<T, 4> make_array(const T& v0, const T& v1, const T& v2, const T& v3)
//{
//    fixed_array<T, 4> v;
//    v[0] = v0;
//    v[1] = v1;
//    v[2] = v2;
//    v[3] = v3;
//    return v;
//}
//
//template<class T>
//inline fixed_array<T, 5> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4)
//{
//    fixed_array<T, 5> v;
//    v[0] = v0;
//    v[1] = v1;
//    v[2] = v2;
//    v[3] = v3;
//    v[4] = v4;
//    return v;
//}
//
//template<class T>
//inline fixed_array<T, 6> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5)
//{
//    fixed_array<T, 6> v;
//    v[0] = v0;
//    v[1] = v1;
//    v[2] = v2;
//    v[3] = v3;
//    v[4] = v4;
//    v[5] = v5;
//    return v;
//}
//
//template<class T>
//inline fixed_array<T, 7> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6)
//{
//    fixed_array<T, 7> v;
//    v[0] = v0;
//    v[1] = v1;
//    v[2] = v2;
//    v[3] = v3;
//    v[4] = v4;
//    v[5] = v5;
//    v[6] = v6;
//    return v;
//}
//
//template<class T>
//inline fixed_array<T, 8> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7)
//{
//    fixed_array<T, 8> v;
//    v[0] = v0;
//    v[1] = v1;
//    v[2] = v2;
//    v[3] = v3;
//    v[4] = v4;
//    v[5] = v5;
//    v[6] = v6;
//    v[7] = v7;
//    return v;
//}
//
//template<class T>
//inline fixed_array<T, 9> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7, const T& v8)
//{
//    fixed_array<T, 9> v;
//    v[0] = v0;
//    v[1] = v1;
//    v[2] = v2;
//    v[3] = v3;
//    v[4] = v4;
//    v[5] = v5;
//    v[6] = v6;
//    v[7] = v7;
//    v[8] = v8;
//    return v;
//}
//
//template<class T>
//inline fixed_array<T, 10> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7, const T& v8, const T& v9)
//{
//    fixed_array<T, 10> v;
//    v[0] = v0;
//    v[1] = v1;
//    v[2] = v2;
//    v[3] = v3;
//    v[4] = v4;
//    v[5] = v5;
//    v[6] = v6;
//    v[7] = v7;
//    v[8] = v8;
//    v[9] = v9;
//    return v;
//}

#ifndef FIXED_ARRAY_CPP

//extern template class SOFA_HELPER_API fixed_array<float, 1>;
//extern template class SOFA_HELPER_API fixed_array<double, 1>;

extern template class SOFA_HELPER_API std::array<float, 2>;
extern template class SOFA_HELPER_API std::array<double, 2>;

extern template class SOFA_HELPER_API std::array<float, 3>;
extern template class SOFA_HELPER_API std::array<double, 3>;

extern template class SOFA_HELPER_API std::array<float, 4>;
extern template class SOFA_HELPER_API std::array<double, 4>;

extern template class SOFA_HELPER_API std::array<float, 5>;
extern template class SOFA_HELPER_API std::array<double, 5>;

extern template class SOFA_HELPER_API std::array<float, 6>;
extern template class SOFA_HELPER_API std::array<double, 6>;

extern template class SOFA_HELPER_API std::array<float, 7>;
extern template class SOFA_HELPER_API std::array<double, 7>;
#endif //

} // namespace helper

} // namespace sofa

namespace std
{
    template <typename T, size_t N>
    std::ostream& operator << (std::ostream& out, const std::array<T, N>& a)
    {
        static_assert(N > 0, "Cannot create a zero size arrays");
        for (auto i = 0; i < N - 1; i++)
            out << a[i] << " ";
        out << a[N - 1];
        return out;
    }

    template <typename T, size_t N>
    std::istream& operator >> (std::istream& in, std::array<T, N>& a)
    {
        for (auto i = 0; i < N; i++)
            in >> a[i];
        return in;
    }
} // namespace std

#endif
