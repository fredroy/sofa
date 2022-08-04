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
#ifndef SOFA_CORE_VECID_H
#define SOFA_CORE_VECID_H

#include <sofa/config.h>
#include <string>
#include <sstream>
#include <cassert>
#include <unordered_map>
#include <array>
namespace sofa::core
{

/// Types of vectors that can be stored in State
enum VecType
{
    V_ALL = 0,
    V_COORD,
    V_DERIV,
    V_MATDERIV,
};

static constexpr std::array< const char*, 4> VecTypeLabels
{
        "(V_ALL)",
        "(V_COORD)",
        "(V_DERIV)",
        "(V_MATDERIV)"
};

//static const std::unordered_map<VecType, std::string> VecTypeLabels {
//    {V_ALL, "(V_ALL)"},
//    {V_COORD, "(V_COORD)"},
//    {V_DERIV, "(V_DERIV)"},
//    {V_MATDERIV, "(V_MATDERIV)"}
//};

/// Types of vectors that can be stored in State
enum VecAccess
{
    V_READ=0,
    V_WRITE,
};

template <VecType vtype, VecAccess vaccess>
class TVecId;

template <VecType vtype, VecAccess vaccess>
class TStandardVec;


template <VecAccess vaccess>
class TStandardVec<V_COORD, vaccess>
{
public:
    typedef TVecId<V_COORD, vaccess> MyVecId;
    static constexpr MyVecId position()      { return MyVecId(1);}
    static constexpr MyVecId restPosition()  { return MyVecId(2);}
    static constexpr MyVecId freePosition()  { return MyVecId(3);}
    static constexpr MyVecId resetPosition() { return MyVecId(4);}
    static constexpr const sofa::Size V_FIRST_DYNAMIC_INDEX = 5; ///< This is the first index used for dynamically allocated vectors

    static std::string getName(const MyVecId& v)
    {
        std::string result;
        switch(v.getIndex())
        {
        case 0: result+= "null";
            break;
        case 1: result+= "position";
            break;
        case 2: result+= "restPosition";
            break;
        case 3: result+= "freePosition";
            break;
        case 4: result+= "resetPosition";
            break;
        default:
            std::ostringstream out;
            out << v.getIndex();
            result+= out.str();
            break;
        }
        result+= VecTypeLabels.at(V_COORD);
        return result;
    }

    static constexpr const char* getGroup(const MyVecId& v)
    {
        switch(v.getIndex())
        {
            case 0: return {}; //null
            case 1: return "States"; //position
            case 2: return "Rest States"; //restPosition
            case 3: return "Free Motion"; //freePosition
            case 4: return "States"; //
            default: return "";
        }
    }
};

template <VecAccess vaccess>
class TStandardVec<V_DERIV, vaccess>
{
public:
    typedef TVecId<V_DERIV, vaccess> MyVecId;

    static constexpr MyVecId velocity()       { return MyVecId(1); }
    static constexpr MyVecId resetVelocity()  { return MyVecId(2); }
    static constexpr MyVecId freeVelocity()   { return MyVecId(3); }
    static constexpr MyVecId normal()         { return MyVecId(4); }
    static constexpr MyVecId force()          { return MyVecId(5); }
    static constexpr MyVecId externalForce()  { return MyVecId(6); }
    static constexpr MyVecId dx()             { return MyVecId(7); }
    static constexpr MyVecId dforce()         { return MyVecId(8); }
    static constexpr const sofa::Size V_FIRST_DYNAMIC_INDEX = 9; ///< This is the first index used for dynamically allocated vectors

    static std::string getName(const MyVecId& v)
    {
        std::string result;
        switch(v.getIndex())
        {
        case 0: result+= "null";
            break;
        case 1: result+= "velocity";
            break;
        case 2: result+= "resetVelocity";
            break;
        case 3: result+= "freeVelocity";
            break;
        case 4: result+= "normal";
            break;
        case 5: result+= "force";
            break;
        case 6: result+= "externalForce";
            break;
        case 7: result+= "dx";
            break;
        case 8: result+= "dforce";
            break;
        default:
            std::ostringstream out;
            out << v.getIndex();
            result+= out.str();
            break;
        }
        result+= VecTypeLabels.at(V_DERIV);
        return result.c_str();
    }

    static constexpr const char* getGroup(const MyVecId& v)
    {
        switch(v.getIndex())
        {
            case 0: return {}; //null
            case 1: return "States"; //velocity
            case 2: return "States"; //resetVelocity
            case 3: return "Free Motion"; //freeVelocity
            case 4: return "States"; //normal
            case 5: return "Force"; //force
            case 6: return "Force"; //externalForce
            case 7: return "States"; //dx
            case 8: return "Force"; //dforce
            default: return "";
        }
    }
};

template <VecAccess vaccess>
class TStandardVec<V_MATDERIV, vaccess>
{
public:
    typedef TVecId<V_MATDERIV, vaccess> MyVecId;

    static constexpr MyVecId constraintJacobian()    { return MyVecId(1);} // jacobian matrix of constraints
    static constexpr MyVecId mappingJacobian() { return MyVecId(2);}         // accumulated matrix of the mappings

    SOFA_ATTRIBUTE_DISABLED("v17.06 (PR#276)", "v21.06", "See VecId.h to remove this message and replace by constraintMatrix")
    static MyVecId holonomicC() = delete;
    static constexpr const sofa::Size V_FIRST_DYNAMIC_INDEX = 3; ///< This is the first index used for dynamically allocated vectors

    static std::string getName(const MyVecId& v)
    {
        std::string result;
        switch(v.getIndex())
        {
        case 0: result+= "null";
            break;
        case 1: result+= "holonomic";
            break;
        case 2: result+= "nonHolonomic";
            break;
        default:
            std::ostringstream out;
            out << v.getIndex();
            result+= out.str();
            break;
        }
        result+= "(V_MATDERIV)";
        return result.c_str();
    }

    static std::string getGroup(const MyVecId& v)
    {
        switch(v.getIndex())
        {
        case 0: return {}; //null
            case 1: return "Jacobian"; //constraintJacobian
            case 2: return "Jacobian"; //mappingJacobian
            default: return {};
        }
    }
};

template <VecAccess vaccess>
class TStandardVec<V_ALL, vaccess>
    : public TStandardVec<V_COORD,vaccess>
    , public TStandardVec<V_DERIV,vaccess>
    , public TStandardVec<V_MATDERIV,vaccess>
{
public:
    typedef TVecId<V_ALL, vaccess> MyVecId;

    static constexpr sofa::Index getFirstDynamicIndex(VecType t)
    {
        switch(t)
        {
        case V_COORD:
            return TStandardVec<V_COORD,vaccess>::V_FIRST_DYNAMIC_INDEX;
        case V_DERIV:
            return TStandardVec<V_DERIV,vaccess>::V_FIRST_DYNAMIC_INDEX;
        case V_MATDERIV:
            return TStandardVec<V_MATDERIV,vaccess>::V_FIRST_DYNAMIC_INDEX;
        default:
            return 0;
        }
    }

    static std::string getName(const MyVecId& v)
    {
        switch(v.getType())
        {
        case V_COORD:
            return TStandardVec<V_COORD,vaccess>::getName(static_cast<TVecId<V_COORD,vaccess>>(v));
        case V_DERIV:
            return TStandardVec<V_DERIV,vaccess>::getName(static_cast<TVecId<V_DERIV,vaccess>>(v));
        case V_MATDERIV:
            return TStandardVec<V_MATDERIV,vaccess>::getName(static_cast<TVecId<V_MATDERIV,vaccess>>(v));
        default:
            std::string result;
            std::ostringstream out;
            out << v.getIndex() << "(" << v.getType() << ")";
            result = out.str();
            return result;
        }
    }
};

/// This is a base class for TVecId that contains all the data stored.
///
/// @note TVecId itself stores no data, in order to be able to convert between templates inplace with reinterpret_cast
/// for performance reasons (typically when working with TMultiVecId instances, which would otherwise copy maps of TVecId).
/// This is (a little) less efficient for non V_ALL versions, but is without comparison with the loss of performance
/// with the typical operation of passing a stored "TMultiVecId<!V_ALL,V_WRITE>" to a method taking a "const TMultiVecId<V_ALL,V_READ>&".
class BaseVecId
{
public:
    constexpr VecType getType() const { return type; }
    constexpr sofa::Index getIndex() const { return index; }

	VecType type;
    sofa::Index index;

protected:
    constexpr BaseVecId(VecType t, sofa::Index i) : type(t), index(i) {}
};

/// This class is only here as fix for a VC2010 compiler otherwise padding TVecId<V_ALL,?> with 4 more bytes than TVecId<?,?>, 
/// probably due to some weird rule requiring to have distinct base pointers with multiple inheritance that's imo
/// wrongly applied for base classes without data members, and hopefully should not make anything worse for other compilers.
/// @note Just in case, we have a static size assertion at the end of the file, so you will know if there is a problem.
class VecIdAlignFix {};

struct VecIdProperties
{
    std::string label;
    std::string group;
};

/// Identify a vector of a given type stored in State
/// This class is templated in order to create different variations (generic versus specific type, read-only vs write access)
template <VecType vtype, VecAccess vaccess>
class TVecId : public BaseVecId, public TStandardVec<vtype, vaccess>, public VecIdAlignFix
{
public:
    constexpr TVecId() : BaseVecId(vtype, 0) { }
    constexpr TVecId(sofa::Index i) : BaseVecId(vtype, i) { }

    /// Copy constructor
    constexpr TVecId(const TVecId<vtype, vaccess> & v) : BaseVecId(vtype, v.getIndex()) {}

    /// Copy from another VecId, possibly with another type of access, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecAccess vaccess2>
    constexpr TVecId(const TVecId<vtype, vaccess2>& v) : BaseVecId(vtype, v.getIndex())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
    }

	template<VecAccess vaccess2>
    explicit constexpr TVecId(const TVecId<V_ALL, vaccess2>& v) : BaseVecId(vtype, v.getIndex())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
#ifndef NDEBUG
        assert(v.getType() == vtype);
#endif
    }

    // Copy assignment

    constexpr TVecId<vtype, vaccess> & operator=(const TVecId<vtype, vaccess>& other) 
    {
        this->index = other.index;
        this->type = other.type;
        return *this;
    }

    template<VecAccess vaccess2>
    constexpr TVecId<vtype, vaccess> & operator=(const TVecId<vtype, vaccess2>& other) 
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
        this->index = other.index;
        this->type = other.type;
        return *this;
    }

    template<VecAccess vaccess2>
    constexpr TVecId<vtype, vaccess> & operator=(const TVecId<V_ALL, vaccess2>& other) 
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
#ifndef NDEBUG
        assert(other.getType() == vtype);
#endif
        this->index = other.index;
        this->type = other.type;
        return *this;
    }


    template<VecType vtype2, VecAccess vaccess2>
    constexpr bool operator==(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() == v.getType() && getIndex() == v.getIndex();
    }

    template<VecType vtype2, VecAccess vaccess2>
    constexpr bool operator!=(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() != v.getType() || getIndex() != v.getIndex();
    }

    static constexpr TVecId null() { return TVecId(0);}
    constexpr bool isNull() const { return this->index == 0; }

    std::string getName() const
    {
        return TStandardVec<vtype, vaccess>::getName(*this);
    }
    std::string getGroup() const
    {
        return TStandardVec<vtype, vaccess>::getGroup(*this);
    }

    friend inline std::ostream& operator << ( std::ostream& out, const TVecId& v )
    {
        out << v.getName();
        return out;
    }
};

/// Identify any vector stored in State
template<VecAccess vaccess>
class TVecId<V_ALL, vaccess> : public BaseVecId, public TStandardVec<V_ALL, vaccess>
{
public:
    constexpr TVecId() : BaseVecId(V_ALL, 0) { }
    constexpr TVecId(VecType t, sofa::Index i) : BaseVecId(t, i) { }
    /// Create a generic VecId from a specific or generic one, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecType vtype2, VecAccess vaccess2>
    constexpr TVecId(const TVecId<vtype2, vaccess2>& v) : BaseVecId(v.getType(), v.getIndex())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
    }

    // Copy assignment
    template<VecType vtype2, VecAccess vaccess2>
    constexpr TVecId<V_ALL, vaccess> & operator=(const TVecId<vtype2, vaccess2>& other) {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
        this->index = other.index;
        this->type = other.type;
        return *this;
    }


    template<VecType vtype2, VecAccess vaccess2>
    constexpr bool operator==(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() == v.getType() && getIndex() == v.getIndex();
    }

    template<VecType vtype2, VecAccess vaccess2>
    constexpr bool operator!=(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() != v.getType() || getIndex() != v.getIndex();
    }

    static constexpr TVecId null() { return TVecId(V_ALL, 0);}
    constexpr bool isNull() const { return this->index == 0; }

    std::string getName() const
    {
        return TStandardVec<V_ALL, vaccess>::getName(*this);
    }
    std::string getGroup() const
    {
        return TStandardVec<V_ALL, vaccess>::getGroup(*this);
    }
    friend inline std::ostream& operator << ( std::ostream& out, const TVecId& v )
    {
        out << v.getName();
        return out;
    }
};

/// Identify one vector stored in State
/// A ConstVecId only provides a read-only access to the underlying vector.
typedef TVecId<V_ALL, V_READ> ConstVecId;

/// Identify one vector stored in State
/// A VecId provides a read-write access to the underlying vector.
typedef TVecId<V_ALL, V_WRITE> VecId;

/// Typedefs for each type of state vectors
typedef TVecId<V_COORD, V_READ> ConstVecCoordId;
typedef TVecId<V_COORD, V_WRITE>     VecCoordId;
typedef TVecId<V_DERIV, V_READ> ConstVecDerivId;
typedef TVecId<V_DERIV, V_WRITE>     VecDerivId;
typedef TVecId<V_MATDERIV, V_READ> ConstMatrixDerivId;
typedef TVecId<V_MATDERIV, V_WRITE>     MatrixDerivId;

static_assert(sizeof(VecId) == sizeof(VecCoordId), "");
} // namespace sofa

#endif
