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

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <sofa/helper/config.h>

#include <fmt/format.h>

namespace sofa::helper::lifecycle
{

class SOFA_HELPER_API ComponentChange
{
public:
    ComponentChange() = default;
    explicit ComponentChange(const std::string& sinceVersion)
    {
        std::stringstream output;
        output << "This component changed since SOFA " << sinceVersion;
        m_message = output.str();
        m_changeVersion = sinceVersion;
    }
    virtual ~ComponentChange() = default;

    std::string m_message;
    std::string m_changeVersion;
    const std::string& getMessage() const { return m_message; }
    const std::string& getVersion() const { return m_changeVersion; }

    ComponentChange& withCustomMessage(const std::string& message) { m_message = message; return *this; }
};

class SOFA_HELPER_API Deprecated : public ComponentChange
{
public:
    explicit Deprecated(const std::string& sinceVersion, const std::string& untilVersion, const std::string& instruction="\b")
    {
        m_message = fmt::format(R"(
"This component has been DEPRECATED since SOFA {0} and will be removed in SOFA {1}.
{2}
Please consider updating your scene as using deprecated component may result in poor performance and undefined behavior.
If this component is crucial to you please report in a GitHub issue in order to reconsider this component for future re-integration.;
)", sinceVersion, untilVersion, instruction);
        m_changeVersion = untilVersion;
    }
};

class SOFA_HELPER_API Pluginized : public ComponentChange
{
public:
    explicit Pluginized(const std::string& sinceVersion, const std::string& plugin)
    {
        m_message = fmt::format(R"(
"This component has been PLUGINIZED since SOFA {0}.
To continue using this component you need to update your scene and add <RequiredPlugin name='{1}'/>
)", sinceVersion, plugin);
        m_changeVersion = sinceVersion;
    }
};

struct SOFA_HELPER_API RemovedIn
{
    explicit RemovedIn(std::string removalVersion) : m_removalVersion(std::move(removalVersion)) {}

private:
    struct SOFA_HELPER_API AfterDeprecationIn : public ComponentChange
    {
        AfterDeprecationIn(const std::string& removalVersion, const std::string& deprecationVersion)
        {
            m_message = fmt::format(R"(
"This component has been REMOVED since SOFA {0} (deprecated since {1}).
Please consider updating your scene. If this component is crucial to you please report in a GitHub issue in order to reconsider this component for future re-integration.
)", removalVersion, deprecationVersion);
            m_changeVersion = removalVersion;
        }
    };

    struct SOFA_HELPER_API WithoutAnyDeprecation : public ComponentChange
    {
        explicit WithoutAnyDeprecation(const std::string& removalVersion)
        {
            m_message = fmt::format(R"(
"This component has been REMOVED since SOFA {0}.
Please consider updating your scene. If this component is crucial to you please report in a GitHub issue in order to reconsider this component for future re-integration.
)", removalVersion);
            m_changeVersion = removalVersion;
        }
    };

public:
    [[nodiscard]] AfterDeprecationIn afterDeprecationIn(const std::string& deprecationVersion) const
    {
        return { m_removalVersion, deprecationVersion };
    }

    [[nodiscard]] WithoutAnyDeprecation withoutAnyDeprecation() const
    {
        return WithoutAnyDeprecation{ m_removalVersion };
    }

private:
    std::string m_removalVersion;
};

class SOFA_HELPER_API Moved : public ComponentChange
{
public:
    Moved(const std::string& sinceVersion, const std::string& fromPlugin, const std::string& toPlugin)
    {
        m_message = fmt::format(R"(
"This component has been MOVED from {0} to {1} since SOFA {2}.
To continue using this component you may need to update your scene. by adding\n<RequiredPlugin name='{1}'/>
)", fromPlugin, toPlugin, sinceVersion);
        m_changeVersion = sinceVersion;
    }
};

class SOFA_HELPER_API Renamed : public ComponentChange
{
public:
    Renamed(const std::string& sinceVersion, const std::string& untilVersion,  const std::string& newName)
    {
        m_message = fmt::format(R"(
"This component has been RENAMED to {0} since SOFA {1}, and this alias will be removed in SOFA {2}.
To continue using this component after SOFA {2} you will need to update your scene."
)", newName, sinceVersion, untilVersion);
        m_changeVersion = untilVersion;
        m_newName = newName;
    }

   const std::string& getNewName() const
    {
        return m_newName;
    }

private:
    std::string m_newName;
};

class SOFA_HELPER_API Dealiased : public ComponentChange
{
public:
    Dealiased(const std::string& sinceVersion, const std::string& originalName)
    {
        m_message = fmt::format(R"(
"This alias for the component {0} was removed in SOFA {1}."
)", originalName, sinceVersion);
        m_changeVersion = sinceVersion;
        m_originalName = originalName;
    }

    const std::string& getOriginalName() const
    {
        return m_originalName;
    }

private:
    std::string m_originalName;
};

extern SOFA_HELPER_API std::map< std::string, Deprecated, std::less<> > deprecatedComponents;
extern SOFA_HELPER_API std::map< std::string, ComponentChange, std::less<> > movedComponents;
extern SOFA_HELPER_API std::map< std::string, Renamed, std::less<> > renamedComponents;
extern SOFA_HELPER_API std::map< std::string, ComponentChange, std::less<> > uncreatableComponents;
extern SOFA_HELPER_API std::map< std::string, Dealiased, std::less<> > dealiasedComponents;

} // namespace sofa::helper::lifecycle
