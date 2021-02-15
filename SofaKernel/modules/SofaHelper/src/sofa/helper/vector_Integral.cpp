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
#define SOFA_HELPER_VECTOR_INTEGRAL_DEFINITION

#include <sofa/helper/vector_T.inl>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::helper
{

/// Input stream
/// Specialization for reading vectors of int and unsigned int using "A-B" notation for all integers between A and B, optionnally specifying a step using "A-B-step" notation.
template<>
SOFA_HELPER_API std::istream& vector<int>::readFromSofaRepr( std::istream& in )
{
    int t;
    this->clear();
    std::string s;
    std::stringstream msg;
    unsigned int numErrors=0;

    /// Cut the input stream in words using the standard's '<space>' token eparator.
    while(in>>s)
    {
        /// Check if there is the sign '-' in the string s.
        std::string::size_type hyphen = s.find_first_of('-',1);

        /// If there is no '-' in s
        if (hyphen == std::string::npos)
        {
            /// Convert the word into an integer number.
            /// Use strtol because it reports error in case of parsing problem.
            t = getInteger(s, msg, numErrors) ;
            this->push_back(t);
        }

        /// If there is at least one '-'
        else
        {
            int t1,t2,tinc;
            std::string s1(s,0,hyphen);
            t1 = getInteger(s1, msg, numErrors) ;
            std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                t2 = getInteger(s2, msg, numErrors) ;
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2-hyphen-1);
                std::string s3(s,hyphen2+1);
                t2 =  getInteger(s2, msg, numErrors) ;
                tinc =  getInteger(s3, msg, numErrors) ;
                if (tinc == 0)
                {
                    tinc = (t1<t2) ? 1 : -1;
                    msg << "- Increment 0 is replaced by "<< tinc << msgendl;
                }
                if ((t2-t1)*tinc < 0)
                {
                    // increment not of the same sign as t2-t1 : swap t1<->t2
                    t = t1;
                    t1 = t2;
                    t2 = t;
                }
            }

            /// Go in backward order.
            if (tinc < 0)
                for (t=t1; t>=t2; t+=tinc)
                    this->push_back(t);
            /// Go in Forward order
            else
                for (t=t1; t<=t2; t+=tinc)
                    this->push_back(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    if(numErrors!=0)
    {
        msg_warning("vector<int>") << "Unable to parse vector values:" << msgendl
                                   << msg.str() ;
    }
    return in;
}


/// Input stream
/// Specialization for reading vectors of int and unsigned int using "A-B" notation for all integers between A and B
template<>
SOFA_HELPER_API std::istream& vector<unsigned int>::readFromSofaRepr( std::istream& in )
{
    std::stringstream errmsg ;
    unsigned int errcnt = 0 ;
    unsigned int t = 0 ;

    this->clear();
    std::string s;

    while(in>>s)
    {
        std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            t = getUnsignedInteger(s, errmsg, errcnt) ;
            this->push_back(t);
        }
        else
        {
            unsigned int t1,t2;
            int tinc;
            std::string s1(s,0,hyphen);
            t1 = getUnsignedInteger(s1, errmsg, errcnt) ;
            std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                t2 = getUnsignedInteger(s2, errmsg, errcnt);
                tinc = (t1<=t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2-hyphen-1);
                std::string s3(s,hyphen2+1);
                t2 = getUnsignedInteger(s2, errmsg, errcnt);
                tinc = getInteger(s3, errmsg, errcnt);
                if (tinc == 0)
                {
                    tinc = (t1<=t2) ? 1 : -1;
                    errmsg << "- problem while parsing '"<<s<<"': increment is 0. Use " << tinc << " instead." ;
                }
                if (((int)(t2-t1))*tinc < 0)
                {
                    /// increment not of the same sign as t2-t1 : swap t1<->t2
                    t = t1;
                    t1 = t2;
                    t2 = t;
                }
            }
            if (tinc < 0){
                for (t=t1; t>t2; t=t+tinc)
                    this->push_back(t);
                this->push_back(t2);
            } else {
                for (t=t1; t<=t2; t=t+tinc)
                    this->push_back(t);
            }
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    if(errcnt!=0)
    {
        msg_warning("vector<unsigned int>") << "Unable to parse values" << msgendl
                                            << errmsg.str() ;
    }

    return in;
}

/// Output stream
/// Specialization for writing vectors of unsigned char
template<>
SOFA_HELPER_API std::ostream& vector<unsigned char>::write(std::ostream& os) const
{
    if( this->size()>0 )
    {
        for( Size i=0; i<this->size()-1; ++i )
            os<<(int)(*this)[i]<<" ";
        os<<(int)(*this)[this->size()-1];
    }
    return os;
}



/// Input stream
/// Specialization for reading vectors of unsigned char
template<>
SOFA_HELPER_API std::istream& vector<unsigned char>::readFromSofaRepr(std::istream& in)
{
    int t;
    this->clear();
    while(in>>t)
    {
        this->push_back((unsigned char)t);
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

} /// namespace sofa::helper


template class SOFA_HELPER_API sofa::helper::vector<bool>;
template class SOFA_HELPER_API sofa::helper::vector<char>;
template class SOFA_HELPER_API sofa::helper::vector<unsigned char>;
template class SOFA_HELPER_API sofa::helper::vector<int>;
template class SOFA_HELPER_API sofa::helper::vector<unsigned int>;
template class SOFA_HELPER_API sofa::helper::vector<long>;
template class SOFA_HELPER_API sofa::helper::vector<unsigned long>;
template class SOFA_HELPER_API sofa::helper::vector<long long>;
template class SOFA_HELPER_API sofa::helper::vector<unsigned long long>;
