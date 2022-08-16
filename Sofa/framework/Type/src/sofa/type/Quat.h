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
#pragma once

#include <sofa/type/config.h>
#include <sofa/type/fwd.h>

#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>

#include <limits>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <array>

namespace sofa::type
{

template<class Real>
class SOFA_TYPE_API Quat
{
    std::array<Real, 4> _q{};

    typedef type::Vec<3, Real> Vec3;
    typedef type::Mat<3,3, Real> Mat3x3;
    typedef type::Mat<4,4, Real> Mat4x4;

    static constexpr Real quaternion_equality_thresold = static_cast<Real>(1e-6);

public:
    typedef Real value_type;
    typedef sofa::Size Size;

    constexpr Quat()
    {
        clear();
    }

    ~Quat() = default;

    constexpr Quat(Real x, Real y, Real z, Real w)
    {
        set(x, y, z, w);
    }

    template<class Real2>
    constexpr Quat(const Real2 q[])
    {
        for (int i = 0; i < 4; i++)
        {
            _q[i] = Real(q[i]);
        }
    }

    template<class Real2>
    constexpr Quat(const Quat<Real2>& q)
    { 
        for (int i = 0; i < 4; i++)
        {
            _q[i] = Real(q[i]);
        }
    }

    constexpr Quat( const Vec3& axis, Real angle )
    {
        axisToQuat(axis, angle);
    }

    /** Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo.        
        vFrom and vTo are assumed to be normalized.
    */
    constexpr Quat(const Vec3& vFrom, const Vec3& vTo)
    {
        setFromUnitVectors(vFrom, vTo);
    }

    static constexpr Quat identity()
    {
        return Quat(0,0,0,1);
    }

    constexpr void set(Real x, Real y, Real z, Real w)
    {
        _q[0] = x;
        _q[1] = y;
        _q[2] = z;
        _q[3] = w;
    }

    /// Cast into a standard C array of elements.
    constexpr const Real* ptr() const
    {
        return this->_q.data();
    }

    /// Cast into a standard C array of elements.
    constexpr Real* ptr()
    {
        return this->_q.data();
    }

    /// Returns true if norm of Quaternion is one, false otherwise.
    constexpr bool isNormalized() const
    {
        Real mag = (_q[0] * _q[0] + _q[1] * _q[1] + _q[2] * _q[2] + _q[3] * _q[3]);
        Real epsilon = std::numeric_limits<Real>::epsilon();
        return (std::abs(mag - 1.0) < epsilon);
    }

    /// Normalize a quaternion
    constexpr void normalize()
    {
        const Real mag = (_q[0] * _q[0] + _q[1] * _q[1] + _q[2] * _q[2] + _q[3] * _q[3]);
        double epsilon = 1.0e-10;
        if (std::abs(mag - 1.0) > epsilon)
        {
            if (mag != 0)
            {
                const Real sqr = static_cast<Real>(1.0 / sqrt(mag));
                for (auto& i : _q)
                {
                    i *= sqr;
                }
            }
            else
            {
                _q[3] = 1;
            }
        }
    }

    constexpr void clear()
    {
        set(0.0,0.0,0.0,1);
    }

    /// Convert the reference frame orientation into an orientation quaternion

    constexpr void fromFrame(const Vec3& x, const Vec3&y, const Vec3&z)
    {

        Matrix3 R(x, y, z);
        R.transpose();
        this->fromMatrix(R);
    }
    
    /// Convert a rotation matrix into an orientation quaternion
    constexpr void fromMatrix(const Mat3x3 &m)

    {
        Real tr, s;
        tr = m.x().x() + m.y().y() + m.z().z();

        // check the diagonal
        if (tr > 0)
        {
            s = (float)sqrt(tr + 1);
            _q[3] = s * 0.5f; // w OK
            s = 0.5f / s;
            _q[0] = (m.z().y() - m.y().z()) * s; // x OK
            _q[1] = (m.x().z() - m.z().x()) * s; // y OK
            _q[2] = (m.y().x() - m.x().y()) * s; // z OK
        }
        else
        {
            if (m.y().y() > m.x().x() && m.z().z() <= m.y().y())
            {
                s = (Real)sqrt((m.y().y() - (m.z().z() + m.x().x())) + 1.0f);

                _q[1] = s * 0.5f; // y OK

                if (s != 0.0f)
                    s = 0.5f / s;

                _q[2] = (m.y().z() + m.z().y()) * s; // z OK
                _q[0] = (m.x().y() + m.y().x()) * s; // x OK
                _q[3] = (m.x().z() - m.z().x()) * s; // w OK
            }
            else if ((m.y().y() <= m.x().x() && m.z().z() > m.x().x()) || (m.z().z() > m.y().y()))
            {
                s = (Real)sqrt((m.z().z() - (m.x().x() + m.y().y())) + 1.0f);

                _q[2] = s * 0.5f; // z OK

                if (s != 0.0f)
                    s = 0.5f / s;

                _q[0] = (m.z().x() + m.x().z()) * s; // x OK
                _q[1] = (m.y().z() + m.z().y()) * s; // y OK
                _q[3] = (m.y().x() - m.x().y()) * s; // w OK
            }
            else
            {
                s = (Real)sqrt((m.x().x() - (m.y().y() + m.z().z())) + 1.0f);

                _q[0] = s * 0.5f; // x OK

                if (s != 0.0f)
                    s = 0.5f / s;

                _q[1] = (m.x().y() + m.y().x()) * s; // y OK
                _q[2] = (m.z().x() + m.x().z()) * s; // z OK
                _q[3] = (m.z().y() - m.y().z()) * s; // w OK
            }
        }
    }
   
    /// Convert the quaternion into an orientation matrix
    constexpr void toMatrix(Mat3x3 &m) const
    {
        m[0][0] = (1 - 2 * (_q[1] * _q[1] + _q[2] * _q[2]));
        m[0][1] = (2 * (_q[0] * _q[1] - _q[2] * _q[3]));
        m[0][2] = (2 * (_q[2] * _q[0] + _q[1] * _q[3]));

        m[1][0] = (2 * (_q[0] * _q[1] + _q[2] * _q[3]));
        m[1][1] = (1 - 2 * (_q[2] * _q[2] + _q[0] * _q[0]));
        m[1][2] = (2 * (_q[1] * _q[2] - _q[0] * _q[3]));

        m[2][0] = (2 * (_q[2] * _q[0] - _q[1] * _q[3]));
        m[2][1] = (2 * (_q[1] * _q[2] + _q[0] * _q[3]));
        m[2][2] = (1 - 2 * (_q[1] * _q[1] + _q[0] * _q[0]));
    }

    SOFA_ATTRIBUTE_DEPRECATED__QUAT_API("Function toMatrix(mat4x4) will be removed. Use toHomogeneousMatrix() instead")
    void toMatrix(Mat4x4 &m) const { toHomogeneousMatrix(m); }

    /// Convert the quaternion into an orientation homogeneous matrix
    /// The homogeneous part is set to 0,0,0,1
    constexpr void toHomogeneousMatrix(Mat4x4 &m) const
    {
        m[0][0] = (1 - 2 * (_q[1] * _q[1] + _q[2] * _q[2]));
        m[0][1] = (2 * (_q[0] * _q[1] - _q[2] * _q[3]));
        m[0][2] = (2 * (_q[2] * _q[0] + _q[1] * _q[3]));
        m[0][3] = 0;

        m[1][0] = (2 * (_q[0] * _q[1] + _q[2] * _q[3]));
        m[1][1] = (1 - 2 * (_q[2] * _q[2] + _q[0] * _q[0]));
        m[1][2] = (2 * (_q[1] * _q[2] - _q[0] * _q[3]));
        m[1][3] = 0;

        m[2][0] = (2 * (_q[2] * _q[0] - _q[1] * _q[3]));
        m[2][1] = (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
        m[2][2] = (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
        m[2][3] = 0;

        m[3][0] = 0;
        m[3][1] = 0;
        m[3][2] = 0;
        m[3][3] = 1;
    }

    /// Apply the rotation to a given vector
    constexpr auto rotate( const Vec3& v ) const -> Vec3
    {
        const Vec3 qxyz{ _q[0], _q[1] , _q[2] };
        const auto t = qxyz.cross(v) * 2;
        return (v + _q[3] * t + qxyz.cross(t));
    }

    /// Apply the inverse rotation to a given vector
    constexpr auto inverseRotate( const Vec3& v ) const -> Vec3
    {
        const Vec3 qxyz{ -_q[0], -_q[1] , -_q[2] };
        const auto t = qxyz.cross(v) * 2;
        return (v + _q[3] * t + qxyz.cross(t));
    }

    /// Given two quaternions, add them together to get a third quaternion.
    /// Adding quaternions to get a compound rotation is analagous to adding
    /// translations to get a compound translation.
    /// 
    constexpr auto operator+(const Quat &q1) const -> Quat
    {
        //    static int	count	= 0;

        Real		t1[4], t2[4], t3[4];
        Real		tf[4];
        Quat    	ret;

        t1[0] = _q[0] * q1._q[3];
        t1[1] = _q[1] * q1._q[3];
        t1[2] = _q[2] * q1._q[3];

        t2[0] = q1._q[0] * _q[3];
        t2[1] = q1._q[1] * _q[3];
        t2[2] = q1._q[2] * _q[3];

        // cross product t3 = q2 x q1
        t3[0] = (q1._q[1] * _q[2]) - (q1._q[2] * _q[1]);
        t3[1] = (q1._q[2] * _q[0]) - (q1._q[0] * _q[2]);
        t3[2] = (q1._q[0] * _q[1]) - (q1._q[1] * _q[0]);
        // end cross product

        tf[0] = t1[0] + t2[0] + t3[0];
        tf[1] = t1[1] + t2[1] + t3[1];
        tf[2] = t1[2] + t2[2] + t3[2];
        tf[3] = _q[3] * q1._q[3] -
            (_q[0] * q1._q[0] + _q[1] * q1._q[1] + _q[2] * q1._q[2]);

        ret._q[0] = tf[0];
        ret._q[1] = tf[1];
        ret._q[2] = tf[2];
        ret._q[3] = tf[3];

        ret.normalize();

        return ret;
    }

    constexpr auto operator*(const Quat &q1) const -> Quat
    {
        Quat	ret;

        ret._q[3] = _q[3] * q1._q[3] -
            (_q[0] * q1._q[0] +
                _q[1] * q1._q[1] +
                _q[2] * q1._q[2]);
        ret._q[0] = _q[3] * q1._q[0] +
            _q[0] * q1._q[3] +
            _q[1] * q1._q[2] -
            _q[2] * q1._q[1];
        ret._q[1] = _q[3] * q1._q[1] +
            _q[1] * q1._q[3] +
            _q[2] * q1._q[0] -
            _q[0] * q1._q[2];
        ret._q[2] = _q[3] * q1._q[2] +
            _q[2] * q1._q[3] +
            _q[0] * q1._q[1] -
            _q[1] * q1._q[0];

        return ret;
    }

    constexpr auto operator*(const Real &r) const -> Quat
    {
        Quat  ret;
        ret[0] = _q[0] * r;
        ret[1] = _q[1] * r;
        ret[2] = _q[2] * r;
        ret[3] = _q[3] * r;
        return ret;
    }

    constexpr auto operator/(const Real &r) const -> Quat
    {
        Quat  ret;
        ret[0] = _q[0] / r;
        ret[1] = _q[1] / r;
        ret[2] = _q[2] / r;
        ret[3] = _q[3] / r;
        return ret;
    }

    constexpr void operator*=(const Real &r)
    {
        Quat  ret;
        _q[0] *= r;
        _q[1] *= r;
        _q[2] *= r;
        _q[3] *= r;
    }

    constexpr void operator/=(const Real &r)
    {
        Quat ret;
        _q[0] /= r;
        _q[1] /= r;
        _q[2] /= r;
        _q[3] /= r;
    }

    /// Given two Quats, multiply them together to get a third quaternion.
    constexpr auto quatVectMult(const Vec3& vect) const -> Quat
    {
        Quat	ret;
        ret._q[3] = -(_q[0] * vect[0] + _q[1] * vect[1] + _q[2] * vect[2]);
        ret._q[0] = _q[3] * vect[0] + _q[1] * vect[2] - _q[2] * vect[1];
        ret._q[1] = _q[3] * vect[1] + _q[2] * vect[0] - _q[0] * vect[2];
        ret._q[2] = _q[3] * vect[2] + _q[0] * vect[1] - _q[1] * vect[0];

        return ret;
    }

    constexpr auto vectQuatMult(const Vec3& vect) const -> Quat
    {
        Quat ret;
        ret[3] = -(vect[0] * _q[0] + vect[1] * _q[1] + vect[2] * _q[2]);
        ret[0] = vect[0] * _q[3] + vect[1] * _q[2] - vect[2] * _q[1];
        ret[1] = vect[1] * _q[3] + vect[2] * _q[0] - vect[0] * _q[2];
        ret[2] = vect[2] * _q[3] + vect[0] * _q[1] - vect[1] * _q[0];
        return ret;
    }

    constexpr Real& operator[](Size index)
    {
        assert(index < 4);
        return _q[index];
    }

    constexpr const Real& operator[](Size index) const
    {
        assert(index < 4);
        return _q[index];
    }

    constexpr auto inverse() const -> Quat
    {
        Quat	ret;
        Real		norm = sqrt(_q[0] * _q[0] +
            _q[1] * _q[1] +
            _q[2] * _q[2] +
            _q[3] * _q[3]);

        if (norm != 0.0f)
        {
            norm = 1.0f / norm;
            ret._q[3] = _q[3] * norm;
            for (int i = 0; i < 3; i++)
            {
                ret._q[i] = -_q[i] * norm;
            }
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                ret._q[i] = 0.0;
            }
        }

        return ret;
    }

    constexpr auto quatToRotationVector() const -> Vec3
    {
        Quat q = *this;
        q.normalize();

        Real angle;

        if (q[3] < 0)
            q *= -1; // we only work with theta in [0, PI] (i.e. angle in [0, 2*PI])

        Real sin_half_theta; // note that sin(theta/2) == norm of the imaginary part for unit quaternion

        // to avoid numerical instabilities of acos for theta < 5°
        if (q[3] > 0.999) // theta < 5° -> q[3] = cos(theta/2) > 0.999
        {
            sin_half_theta = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
            angle = (Real)(2.0 * asin(sin_half_theta));
        }
        else
        {
            Real half_theta = acos(q[3]);
            sin_half_theta = sin(half_theta);
            angle = 2 * half_theta;
        }

        assert(sin_half_theta >= 0);
        Vec3 rotVector;
        if (sin_half_theta < std::numeric_limits<Real>::epsilon())
            rotVector = Vec3(Real(0), Real(0), Real(0));
        else
            rotVector = Vec3(q[0], q[1], q[2]) / sin_half_theta * angle;

        return rotVector;
    }

    constexpr auto toEulerVector() const -> Vec3
    {
        Quat q = *this;
        q.normalize();

        // Cancel numerical drifting by clamping on [-1 ; 1]
        Real y = std::max(Real(-1.0), std::min(Real(1.0), Real(2.) * (q[3] * q[1] - q[2] * q[0])));

        Vec3 vEuler;
        vEuler[0] = atan2(2 * (q[3] * q[0] + q[1] * q[2]), (1 - 2 * (q[0] * q[0] + q[1] * q[1])));   //roll
        vEuler[1] = asin(y); // pitch
        vEuler[2] = atan2(2 * (q[3] * q[2] + q[0] * q[1]), (1 - 2 * (q[1] * q[1] + q[2] * q[2])));   //yaw
        return vEuler;
    }

    /*! Returns the slerp interpolation of Quaternions \p a and \p b, at time \p t.
     \p t should range in [0,1]. Result is \p a when \p t=0 and \p b when \p t=1.
     When \p allowFlip is \c true (default) the slerp interpolation will always use the "shortest path"
     between the Quaternions' orientations, by "flipping" the source Quaternion if needed (see
     negate()). */
    constexpr void slerp(const Quat& a, const Quat& b, Real t, bool allowFlip=true)
    {
        Real cosAngle = (Real)(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);

        Real c1, c2;
        // Linear interpolation for close orientations
        if ((1.0 - std::abs(cosAngle)) < 0.01)
        {
            c1 = 1.0f - t;
            c2 = t;
        }
        else
        {
            // Spherical interpolation
            Real angle = (Real)acos((Real)std::abs((Real)cosAngle));
            Real sinAngle = (Real)sin((Real)angle);
            c1 = (Real)sin(angle * (1.0f - t)) / sinAngle;
            c2 = (Real)sin(angle * t) / sinAngle;
        }

        // Use the shortest path
        if (allowFlip && (cosAngle < 0.0f))
            c1 = -c1;

        _q[0] = c1 * a[0] + c2 * b[0];
        _q[1] = c1 * a[1] + c2 * b[1];
        _q[2] = c1 * a[2] + c2 * b[2];
        _q[3] = c1 * a[3] + c2 * b[3];
    }

    /// A useful function, builds a rotation matrix in Matrix based on
    /// given quaternion.
    constexpr void buildRotationMatrix(Real m[4][4]) const
    {
        m[0][0] = (1 - 2 * (_q[1] * _q[1] + _q[2] * _q[2]));
        m[0][1] = (2 * (_q[0] * _q[1] - _q[2] * _q[3]));
        m[0][2] = (2 * (_q[2] * _q[0] + _q[1] * _q[3]));
        m[0][3] = 0;

        m[1][0] = (2 * (_q[0] * _q[1] + _q[2] * _q[3]));
        m[1][1] = (1 - 2 * (_q[2] * _q[2] + _q[0] * _q[0]));
        m[1][2] = (2 * (_q[1] * _q[2] - _q[0] * _q[3]));
        m[1][3] = 0;

        m[2][0] = (2 * (_q[2] * _q[0] - _q[1] * _q[3]));
        m[2][1] = (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
        m[2][2] = (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
        m[2][3] = 0;

        m[3][0] = 0;
        m[3][1] = 0;
        m[3][2] = 0;
        m[3][3] = 1;
    }

    constexpr void writeOpenGlMatrix( double* m ) const
    {
        m[0 * 4 + 0] = (double)(1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]));
        m[1 * 4 + 0] = (double)(2.0 * (_q[0] * _q[1] - _q[2] * _q[3]));
        m[2 * 4 + 0] = (double)(2.0 * (_q[2] * _q[0] + _q[1] * _q[3]));
        m[3 * 4 + 0] = (double)0.0;

        m[0 * 4 + 1] = (double)(2.0 * (_q[0] * _q[1] + _q[2] * _q[3]));
        m[1 * 4 + 1] = (double)(1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]));
        m[2 * 4 + 1] = (double)(2.0 * (_q[1] * _q[2] - _q[0] * _q[3]));
        m[3 * 4 + 1] = (double)0.0;

        m[0 * 4 + 2] = (double)(2.0 * (_q[2] * _q[0] - _q[1] * _q[3]));
        m[1 * 4 + 2] = (double)(2.0 * (_q[1] * _q[2] + _q[0] * _q[3]));
        m[2 * 4 + 2] = (double)(1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]));
        m[3 * 4 + 2] = (double)0.0;

        m[0 * 4 + 3] = (double)0.0;
        m[1 * 4 + 3] = (double)0.0;
        m[2 * 4 + 3] = (double)0.0;
        m[3 * 4 + 3] = (double)1.0;
    }

    constexpr void writeOpenGlMatrix( float* m ) const
    {
        m[0 * 4 + 0] = (float)(1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]));
        m[1 * 4 + 0] = (float)(2.0f * (_q[0] * _q[1] - _q[2] * _q[3]));
        m[2 * 4 + 0] = (float)(2.0f * (_q[2] * _q[0] + _q[1] * _q[3]));
        m[3 * 4 + 0] = 0.0f;

        m[0 * 4 + 1] = (float)(2.0f * (_q[0] * _q[1] + _q[2] * _q[3]));
        m[1 * 4 + 1] = (float)(1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]));
        m[2 * 4 + 1] = (float)(2.0f * (_q[1] * _q[2] - _q[0] * _q[3]));
        m[3 * 4 + 1] = 0.0f;

        m[0 * 4 + 2] = (float)(2.0f * (_q[2] * _q[0] - _q[1] * _q[3]));
        m[1 * 4 + 2] = (float)(2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
        m[2 * 4 + 2] = (float)(1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
        m[3 * 4 + 2] = 0.0f;

        m[0 * 4 + 3] = 0.0f;
        m[1 * 4 + 3] = 0.0f;
        m[2 * 4 + 3] = 0.0f;
        m[3 * 4 + 3] = 1.0f;
    }

    /// This function computes a quaternion based on an axis (defined by
    /// the given vector) and an angle about which to rotate.  The angle is
    /// expressed in radians.
    constexpr auto axisToQuat(Vec3 a, Real phi) -> Quat
    {
        if (a.norm() < std::numeric_limits<Real>::epsilon())
        {
            _q[0] = _q[1] = _q[2] = (Real)0.0f;
            _q[3] = (Real)1.0f;

            return Quat();
        }

        a = a / a.norm();
        _q[0] = (Real)a.x();
        _q[1] = (Real)a.y();
        _q[2] = (Real)a.z();

        _q[0] = _q[0] * (Real)sin(phi / 2.0);
        _q[1] = _q[1] * (Real)sin(phi / 2.0);
        _q[2] = _q[2] * (Real)sin(phi / 2.0);

        _q[3] = (Real)cos(phi / 2.0);

        return *this;
    }

    constexpr void quatToAxis(Vec3 & axis, Real &angle) const
    {
        Quat<Real> q = *this;
        if (q[3] < 0)
            q *= -1; // we only work with theta in [0, PI]

        Real sin_half_theta; // note that sin(theta/2) == norm of the imaginary part for unit quaternion

        // to avoid numerical instabilities of acos for theta < 5°
        if (q[3] > 0.999) // theta < 5° -> q[3] = cos(theta/2) > 0.999
        {
            sin_half_theta = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
            angle = (Real)(2.0 * asin(sin_half_theta));
        }
        else
        {
            Real half_theta = acos(q[3]);
            sin_half_theta = sin(half_theta);
            angle = 2 * half_theta;
        }

        assert(sin_half_theta >= 0);
        if (sin_half_theta < std::numeric_limits<Real>::epsilon())
            axis = Vec3(Real(0.), Real(1.), Real(0.));
        else
            axis = Vec3(q[0], q[1], q[2]) / sin_half_theta;
    }

    static constexpr auto createQuaterFromFrame(const Vec3 &lox, const Vec3 &loy,const Vec3 &loz) -> Quat
    {
        Quat q;
        Mat3x3 m;

        for (unsigned int i = 0; i < 3; i++)
        {
            m[i][0] = lox[i];
            m[i][1] = loy[i];
            m[i][2] = loz[i];
        }
        q.fromMatrix(m);
        return q;
    }

    /// Create using rotation vector (axis*angle) given in parent coordinates
    static constexpr auto createFromRotationVector(const Vec3& a) -> Quat
    {
        Real phi = Real(sqrt(a * a));
        if (phi <= 1.0e-5)
            return Quat(0, 0, 0, 1);

        Real nor = 1 / phi;
        Real s = Real(sin(phi / 2));
        return Quat(a[0] * s * nor, a[1] * s * nor, a[2] * s * nor, Real(cos(phi / 2)));
    }

    /// Create a quaternion from Euler angles
    /// Thanks to https://github.com/mrdoob/three.js/blob/dev/src/math/Quaternion.js#L199
    enum class EulerOrder
    {
        XYZ, YXZ, ZXY, ZYX, YZX, XZY
    };

    static constexpr auto createQuaterFromEuler(const Vec3& v, EulerOrder order = EulerOrder::ZYX) -> Quat
    {
        Real quat[4];

        Real c1 = cos(v.elems[0] / 2);
        Real c2 = cos(v.elems[1] / 2);
        Real c3 = cos(v.elems[2] / 2);

        Real s1 = sin(v.elems[0] / 2);
        Real s2 = sin(v.elems[1] / 2);
        Real s3 = sin(v.elems[2] / 2);

        switch (order)
        {
        case EulerOrder::XYZ:
            quat[0] = s1 * c2 * c3 + c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 - s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 + s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 - s1 * s2 * s3;
            break;
        case EulerOrder::YXZ:
            quat[0] = s1 * c2 * c3 + c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 - s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 - s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 + s1 * s2 * s3;
            break;
        case EulerOrder::ZXY:
            quat[0] = s1 * c2 * c3 - c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 + s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 + s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 - s1 * s2 * s3;
            break;
        case EulerOrder::YZX:
            quat[0] = s1 * c2 * c3 + c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 + s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 - s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 - s1 * s2 * s3;
            break;
        case EulerOrder::XZY:
            quat[0] = s1 * c2 * c3 - c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 - s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 + s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 + s1 * s2 * s3;
            break;
        default:
        case EulerOrder::ZYX:
            quat[0] = s1 * c2 * c3 - c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 + s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 - s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 + s1 * s2 * s3;
            break;
        }

        Quat quatResult{ quat[0], quat[1], quat[2], quat[3] };
        return quatResult;
    }

    /// Create a quaternion from Euler angles
    static constexpr auto fromEuler( Real alpha, Real beta, Real gamma, EulerOrder order = EulerOrder::ZYX ) -> Quat
    {
        return createQuaterFromEuler({ alpha, beta, gamma }, order);
    }

    /// Create using the entries of a rotation vector (axis*angle) given in parent coordinates
    static constexpr auto createFromRotationVector(Real a0, Real a1, Real a2 ) -> Quat
    {
        Real phi = Real(sqrt(a0 * a0 + a1 * a1 + a2 * a2));

        if (phi >= 1.0e-5)
            return Quat(0, 0, 0, 1);

        Real nor = 1 / phi;
        Real s = sin(phi / Real(2.));
        return Quat(a0 * s * nor, a1 * s * nor, a2 * s * nor, cos(phi / Real(2.)));
    }

    /// Create using rotation vector (axis*angle) given in parent coordinates
    static constexpr auto set(const Vec3& a) { return createFromRotationVector(a); }

    /// Create using using the entries of a rotation vector (axis*angle) given in parent coordinates
    static constexpr auto set(Real a0, Real a1, Real a2) { return createFromRotationVector(a0,a1,a2); }

    /// Return the quaternion resulting of the movement between 2 quaternions
    static constexpr auto quatDiff( Quat a, const Quat& b) -> Quat
    {
        // If the axes are not oriented in the same direction, flip the axis and angle of a to get the same convention than b
        if (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0)
        {
            a[0] = -a[0];
            a[1] = -a[1];
            a[2] = -a[2];
            a[3] = -a[3];
        }

        Quat q = b.inverse() * a;
        return q;
    }

    /// Return the eulerian vector resulting of the movement between 2 quaternions
    static constexpr auto angularDisplacement( const Quat& a, const Quat& b) -> Vec3
    {
        // In the following, use of quatToRotationVector instead of toEulerVector:
        // this is done to keep the old behavior (before the correction of the toEulerVector function).
        return quatDiff(a, b).quatToRotationVector();
    }

    /// Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo. vFrom and vTo are assumed to be normalized.
    constexpr void setFromUnitVectors(const Vec3& vFrom, const Vec3& vTo)
    {
        Vec3 v1;
        Real epsilon = Real(0.0001);

        Real res_dot = type::dot(vFrom, vTo) + 1;
        if (res_dot < epsilon)
        {
            res_dot = 0;
            if (fabs(vFrom[0]) > fabs(vFrom[2]))
                v1 = Vec3(-vFrom[1], vFrom[0], Real(0.));
            else
                v1 = Vec3(Real(0.), -vFrom[2], vFrom[1]);
        }
        else
        {
            v1 = vFrom.cross(vTo);
        }

        _q[0] = v1[0];
        _q[1] = v1[1];
        _q[2] = v1[2];
        _q[3] = res_dot;

        this->normalize();
    }

    SOFA_ATTRIBUTE_DEPRECATED__QUAT_API("This function will be removed. use iostream operators instead.")
    void print()
    {
        printf("(%f, %f ,%f, %f)\n", _q[0], _q[1], _q[2], _q[3]);
    }

    constexpr auto slerp(const Quat &q1, Real t) const -> Quat
    {
        Quat q0_1;
        for (unsigned int i = 0; i < 3; i++)
            q0_1[i] = -_q[i];

        q0_1[3] = _q[3];

        q0_1 = q1 * q0_1;

        Vec3 axis, temp;
        Real angle;

        q0_1.quatToAxis(axis, angle);

        temp = axis * sin(t * angle);
        for (unsigned int i = 0; i < 3; i++)
            q0_1[i] = temp[i];

        q0_1[3] = cos(t * angle);
        q0_1 = q0_1 * (*this);
        return q0_1;
    }

    constexpr auto slerp2(const Quat &q1, Real t) const-> Quat
    {
        // quaternion to return
        Quat qm;

        // Calculate angle between them.
        double cosHalfTheta = _q[3] * q1[3] + _q[0] * q1[0] + _q[1] * q1[1] + _q[2] * q1[2];
        // if qa=qb or qa=-qb then theta = 0 and we can return qa
        if (std::abs(cosHalfTheta) >= 1.0)
        {
            qm[3] = _q[3]; qm[0] = _q[0]; qm[1] = _q[1]; qm[2] = _q[2];
            return qm;
        }
        // Calculate temporary values.
        double halfTheta = acos(cosHalfTheta);
        double sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);
        // if theta = 180 degrees then result is not fully defined
        // we could rotate around any axis normal to qa or qb
        if (std::abs(sinHalfTheta) < 0.001)
        {
            qm[3] = (Real)(_q[3] * 0.5 + q1[3] * 0.5);
            qm[0] = (Real)(_q[0] * 0.5 + q1[0] * 0.5);
            qm[1] = (Real)(_q[1] * 0.5 + q1[1] * 0.5);
            qm[2] = (Real)(_q[2] * 0.5 + q1[2] * 0.5);
            return qm;
        }
        double ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
        double ratioB = sin(t * halfTheta) / sinHalfTheta;
        //calculate Quatnion.
        qm[3] = (Real)(_q[3] * ratioA + q1[3] * ratioB);
        qm[0] = (Real)(_q[0] * ratioA + q1[0] * ratioB);
        qm[1] = (Real)(_q[1] * ratioA + q1[1] * ratioB);
        qm[2] = (Real)(_q[2] * ratioA + q1[2] * ratioB);
        return qm;

    }

    constexpr void operator+=(const Quat& q2)
    {
        Real t1[4], t2[4], t3[4];
        Quat q1 = (*this);
        t1[0] = q1._q[0] * q2._q[3];
        t1[1] = q1._q[1] * q2._q[3];
        t1[2] = q1._q[2] * q2._q[3];

        t2[0] = q2._q[0] * q1._q[3];
        t2[1] = q2._q[1] * q1._q[3];
        t2[2] = q2._q[2] * q1._q[3];

        // cross product t3 = q2 x q1
        t3[0] = (q2._q[1] * q1._q[2]) - (q2._q[2] * q1._q[1]);
        t3[1] = (q2._q[2] * q1._q[0]) - (q2._q[0] * q1._q[2]);
        t3[2] = (q2._q[0] * q1._q[1]) - (q2._q[1] * q1._q[0]);
        // end cross product

        _q[0] = t1[0] + t2[0] + t3[0];
        _q[1] = t1[1] + t2[1] + t3[1];
        _q[2] = t1[2] + t2[2] + t3[2];
        _q[3] = q1._q[3] * q2._q[3] -
            (q1._q[0] * q2._q[0] + q1._q[1] * q2._q[1] + q1._q[2] * q2._q[2]);

        normalize();
    }

    constexpr void operator*=(const Quat& q1)
    {
        Quat q2 = *this;
        _q[3] = q2._q[3] * q1._q[3] -
            (q2._q[0] * q1._q[0] +
                q2._q[1] * q1._q[1] +
                q2._q[2] * q1._q[2]);
        _q[0] = q2._q[3] * q1._q[0] +
            q2._q[0] * q1._q[3] +
            q2._q[1] * q1._q[2] -
            q2._q[2] * q1._q[1];
        _q[1] = q2._q[3] * q1._q[1] +
            q2._q[1] * q1._q[3] +
            q2._q[2] * q1._q[0] -
            q2._q[0] * q1._q[2];
        _q[2] = q2._q[3] * q1._q[2] +
            q2._q[2] * q1._q[3] +
            q2._q[0] * q1._q[1] -
            q2._q[1] * q1._q[0];
    }

    constexpr bool operator==(const Quat& q) const
    {
        for (int i = 0; i < 4; i++)
            if (std::abs(_q[i] - q._q[i]) > quaternion_equality_thresold) return false;
        return true;
    }

    constexpr bool operator!=(const Quat& q) const
    {
        for (int i = 0; i < 4; i++)
            if (std::abs(_q[i] - q._q[i]) > quaternion_equality_thresold) return true;
        return false;
    }

    static constexpr Size static_size = 4;
    static constexpr Size size() {return static_size;}

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    static constexpr Size total_size = 4;

    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for quaternions)
    static constexpr Size spatial_dimensions = 3;
};

/// write to an output stream
template<class Real> std::ostream& operator << (std::ostream& out, const Quat<Real>& v);

/// read from an input stream
template<class Real> std::istream& operator >> (std::istream& in, Quat<Real>& v);

} // namespace sofa::type
