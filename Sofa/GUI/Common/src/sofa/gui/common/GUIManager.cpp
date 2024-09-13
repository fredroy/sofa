/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/gui/common/GUIManager.h>

#include <sofa/gui/common/BaseGUI.h>
#include <sofa/gui/common/ArgumentParser.h>

#include <sofa/helper/Utils.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/common/init.h>

#include <fstream>

using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

namespace sofa::gui::common
{

void GUIManager::RegisterParameters(ArgumentParser* argumentParser)
{
    for (const auto& guiCreator : guiCreators)
    {
        if (guiCreator.parameters)
        {
            guiCreator.parameters(argumentParser);
        }
    }
}

int GUIManager::RegisterGUI(const char* name, CreateGUIFn* creator, RegisterGUIParameters* parameters, int priority)
{
    if(guiCreators.size())
    {
        std::list<GUICreator>::iterator it = guiCreators.begin();
        const std::list<GUICreator>::iterator itend = guiCreators.end();
        while (it != itend && strcmp(name, it->name))
            ++it;
        if (it != itend)
        {
            msg_error("GUIManager") << "GUI "<<name<<" duplicate registration.";
            return 1;
        }
    }

    GUICreator entry;
    entry.name = name;
    entry.creator = creator;
    entry.parameters = parameters;
    entry.priority = priority;
    guiCreators.push_back(entry);

    msg_info("GUIManager") << "Registered " << entry.name << " as a GUI.";
    return 0;
}

std::vector<std::string> GUIManager::ListSupportedGUI()
{
    std::vector<std::string> names;
    names.reserve(guiCreators.size());
    for (const auto& guiCreator : guiCreators)
    {
        names.emplace_back(guiCreator.name);
    }
    return names;
}

std::string GUIManager::ListSupportedGUI(const char separator)
{
    const auto guis = GUIManager::ListSupportedGUI();
    return sofa::helper::join(guis.begin(), guis.end(), separator);
}

const char* GUIManager::GetValidGUIName()
{
    const char* name;
    const std::string lastGuiFilename = BaseGUI::getConfigDirectoryPath() + "/lastUsedGUI.ini";
    if (guiCreators.empty())
    {
        msg_error("GUIManager") << "No GUI registered.";
        return nullptr;
    }
    else
    {
        //Check the config file for the last used GUI type
        if(FileSystem::exists(lastGuiFilename))
        {
            std::string lastGuiName;
            std::ifstream lastGuiStream(lastGuiFilename.c_str());
            std::getline(lastGuiStream,lastGuiName);
            lastGuiStream.close();

            const char* lastGuiNameChar = lastGuiName.c_str();

            // const char* lastGuiNameChar = "qt";
            std::list<GUICreator>::iterator it1 = guiCreators.begin();
            const std::list<GUICreator>::iterator itend1 = guiCreators.end();
            while(++it1 != itend1)
            {
                if( strcmp(lastGuiNameChar, it1->name) == 0 )
                {
                    return it1->name;
                }
            }
            msg_warning("GUIManager") << "Previously used GUI not registered. Using default GUI.";
        }
        else
        {
            msg_info("GUIManager") << "lastUsedGUI.ini not found; using default GUI.";
        }

        std::list<GUICreator>::iterator it =guiCreators.begin();
        const std::list<GUICreator>::iterator itend =guiCreators.end();
        name = it->name;
        int prio = it->priority;
        while (++it != itend)
        {
            if (it->priority > prio)
            {
                name = it->name;
                prio = it->priority;
            }
        }
    }
    return name;
}

GUIManager::GUICreator* GUIManager::GetGUICreator(const char* name)
{
    if (!name) name = GetValidGUIName();
    std::list<GUICreator>::iterator it =guiCreators.begin();
    const std::list<GUICreator>::iterator itend =guiCreators.end();
    while (it != itend && strcmp(name, it->name))
        ++it;
    if (it == itend)
    {
        msg_error("GUIManager") << "GUI '"<<name<<"' creation failed."<< msgendl
                                << "Available GUIs: {" << ListSupportedGUI(' ') <<  "}";
        return nullptr;
    }
    else
        return &(*it);
}

int GUIManager::Init(const char* argv0)
{
    BaseGUI::SetProgramName(argv0);

    if (guiCreators.empty())
    {
        msg_error("GUIManager") << "No GUI registered.";
        return 1;
    }

    return 0;
}


sofa::gui::common::BaseGUI* GUIManager::createGUI(const char* name, sofa::simulation::Node::SPtr groot, const char* filename, ArgumentParser* args)
{
    std::string guiName = name;
    
    if (name == nullptr || strcmp(name, "") == 0)
    {
        guiName = GetValidGUIName(); // get the default gui name
    }

    const GUICreator* creator = GetGUICreator(guiName.c_str());
    if (!creator)
    {
        return nullptr;
    }

    sofa::gui::common::BaseGUI* currentGUI = (*creator->creator)(guiName.c_str(), groot, filename, args);
    if (!currentGUI)
    {
        msg_error("GUIManager") << "GUI '"<< guiName <<"' creation failed." ;
        return nullptr;
    }

    if (currentGUI->canBeDefaultGUI())
    {
        //Save this GUI type as the last used GUI
        const std::string lastGuiFilePath = BaseGUI::getConfigDirectoryPath() + "/lastUsedGUI.ini";
        std::ofstream out(lastGuiFilePath.c_str(),std::ios::out);
        out << guiName << std::endl;
        out.close();
    }
    return currentGUI;
}

} // namespace sofa::gui::common
