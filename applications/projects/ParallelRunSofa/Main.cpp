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
#include <sofa/simulation/Node.h>

#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/config.h> // #defines SOFA_HAVE_DAG (or not)
#include <sofa/simulation/common/init.h>
#include <sofa/simulation/graph/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/simulation/SceneLoaderFactory.h>

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/cast.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/Utils.h>

using sofa::core::ExecParams ;

#include <sofa/helper/system/console.h>
using sofa::helper::Utils;

using sofa::simulation::graph::DAGSimulation;
using sofa::helper::system::SetDirectory;
using sofa::core::objectmodel::BaseNode ;

#include <sofa/gui/batch/init.h>

#include <sofa/helper/logging/ConsoleMessageHandler.h>
using sofa::helper::logging::ConsoleMessageHandler ;

#include <sofa/core/logging/RichConsoleStyleMessageFormatter.h>
using  sofa::helper::logging::RichConsoleStyleMessageFormatter ;

#include <sofa/core/logging/PerComponentLoggingMessageHandler.h>
using  sofa::helper::logging::MainPerComponentLoggingMessageHandler ;

#include <sofa/helper/AdvancedTimer.h>

#include <sofa/gui/common/GuiDataRepository.h>
using sofa::gui::common::GuiDataRepository ;

using sofa::helper::system::DataRepository;
using sofa::helper::system::PluginRepository;
using sofa::helper::system::PluginManager;

#include <sofa/helper/logging/MessageDispatcher.h>
using sofa::helper::logging::MessageDispatcher ;

#include <sofa/helper/logging/ClangMessageHandler.h>
using sofa::helper::logging::ClangMessageHandler ;

#include <sofa/helper/logging/ExceptionMessageHandler.h>
using sofa::helper::logging::ExceptionMessageHandler;

#include <sofa/gui/common/ArgumentParser.h>

#include <execution>
#include <thread>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/DefaultTaskScheduler.h>
#include <sofa/component/collision/detection/intersection/MeshMinProximityIntersection.h>
#include <sofa/simulation/WorkerThread.h>
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Add resources dir to GuiDataRepository
    const std::string runSofaIniFilePath = Utils::getSofaPathTo("/etc/runSofa.ini");
    std::map<std::string, std::string> iniFileValues = Utils::readBasicIniFile(runSofaIniFilePath);
    if (iniFileValues.find("RESOURCES_DIR") != iniFileValues.end())
    {
        std::string dir = iniFileValues["RESOURCES_DIR"];
        dir = SetDirectory::GetRelativeFromProcess(dir.c_str());
        if(FileSystem::isDirectory(dir))
        {
            sofa::gui::common::GuiDataRepository.addFirstPath(dir);
        }
    }

    sofa::helper::BackTrace::autodump();

#ifdef WIN32
    {
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        COORD s;
        s.X = 160; s.Y = 10000;
        SetConsoleScreenBufferSize(hStdout, s);
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(hStdout, &csbi))
        {
            SMALL_RECT winfo;
            winfo = csbi.srWindow;
            //winfo.Top = 0;
            winfo.Left = 0;
            //winfo.Bottom = csbi.dwSize.Y-1;
            winfo.Right = csbi.dwMaximumWindowSize.X-1;
            SetConsoleWindowInfo(hStdout, TRUE, &winfo);
        }

    }
#endif

    std::string fileName ;
    bool        startAnim = false;
    bool        showHelp = false;
    bool        printFactory = false;
    bool        loadRecent = false;
    bool        temporaryFile = false;
    bool        testMode = false;
    bool        noAutoloadPlugins = false;
    bool        noSceneCheck = false;
    unsigned int nbMSSASamples = 1;
    bool computationTimeAtBegin = false;
    unsigned int computationTimeSampling=0; ///< Frequency of display of the computation time statistics, in number of animation steps. 0 means never.
    std::string    computationTimeOutputType="stdout";

    std::string gui = "";
    std::string verif = "";

#if defined(SOFA_HAVE_DAG)
    std::string simulationType = "dag";
#else
    std::string simulationType = "tree";
#endif

    std::vector<std::string> plugins;
    std::vector<std::string> files;

    std::string colorsStatus = "unset";
    std::string messageHandler = "auto";
    bool enableInteraction = false ;
    int width = 800;
    int height = 600;

    // Argument parser has 2 stages
    // one is for the runSofa options itself
    // second is for the eventual options the GUIs can add (i.e batch with the "-n" number of iterations option) 
    sofa::gui::common::ArgumentParser* argParser = new sofa::gui::common::ArgumentParser(argc, argv);

    argParser->addArgument(
        cxxopts::value<std::vector<std::string>>(plugins),
        "l,load",
        "load given plugins"
    );
        
    argParser->addArgument(
        cxxopts::value<std::string>(messageHandler)
        ->default_value("auto"),
        "f,formatting",
        "select the message formatting to use (auto, clang, sofa, rich, test)"
    );
    
    // first option parsing to see if the user requested to show help
    argParser->parse();

    if(showHelp)
    {
        argParser->showHelp();
        exit( EXIT_SUCCESS );
    }

    // Note that initializations must be done after ArgumentParser that can exit the application (without cleanup)
    // even if everything is ok e.g. asking for help
    sofa::simulation::graph::init();

    if (simulationType == "tree")
        msg_warning("runSofa") << "Tree based simulation, switching back to graph simulation.";
    assert(sofa::simulation::getSimulation());

    if (colorsStatus == "unset") {
        // If the parameter is unset, check the environment variable
        const char * colorStatusEnvironment = std::getenv("SOFA_COLOR_TERMINAL");
        if (colorStatusEnvironment != nullptr) {
            const std::string status (colorStatusEnvironment);
            if (status == "yes" || status == "on" || status == "always")
                sofa::helper::console::setStatus(sofa::helper::console::Status::On);
            else if (status == "no" || status == "off" || status == "never")
                sofa::helper::console::setStatus(sofa::helper::console::Status::Off);
            else
                sofa::helper::console::setStatus(sofa::helper::console::Status::Auto);
        }
    } else if (colorsStatus == "auto")
        sofa::helper::console::setStatus(sofa::helper::console::Status::Auto);
    else if (colorsStatus == "yes")
        sofa::helper::console::setStatus(sofa::helper::console::Status::On);
    else if (colorsStatus == "no")
        sofa::helper::console::setStatus(sofa::helper::console::Status::Off);

    //TODO(dmarchal): Use smart pointer there to avoid memory leaks !!
    if (messageHandler == "auto" )
    {
        MessageDispatcher::clearHandlers() ;
        MessageDispatcher::addHandler( new ConsoleMessageHandler() ) ;
    }
    else if (messageHandler == "clang")
    {
        MessageDispatcher::clearHandlers() ;
        MessageDispatcher::addHandler( new ClangMessageHandler() ) ;
    }
    else if (messageHandler == "sofa")
    {
        MessageDispatcher::clearHandlers() ;
        MessageDispatcher::addHandler( new ConsoleMessageHandler() ) ;
    }
    else if (messageHandler == "rich")
    {
        MessageDispatcher::clearHandlers() ;
        MessageDispatcher::addHandler( new ConsoleMessageHandler(&RichConsoleStyleMessageFormatter::getInstance()) ) ;
    }
    else if (messageHandler == "test"){
        MessageDispatcher::addHandler( new ExceptionMessageHandler() ) ;
    }
    else{
        msg_warning("") << "Invalid argument '" << messageHandler << "' for '--formatting'";
    }
    MessageDispatcher::addHandler(&MainPerComponentLoggingMessageHandler::getInstance()) ;
#ifdef TRACY_ENABLE
    MessageDispatcher::addHandler(&sofa::helper::logging::MainTracyMessageHandler::getInstance());
#endif

    // Output FileRepositories
    msg_info("runSofa") << "PluginRepository paths = " << PluginRepository.getPathsJoined();
    msg_info("runSofa") << "DataRepository paths = " << DataRepository.getPathsJoined();
    msg_info("runSofa") << "GuiDataRepository paths = " << GuiDataRepository.getPathsJoined();


    for (const auto& plugin : plugins)
    {
        PluginManager::getInstance().loadPlugin(plugin);
    }

    // Parse again to take into account the potential new options
    argParser->parse();

    // Fetching file name must be done after the additionnal potential options have been added
    // otherwise the first parsing will take the unknown options as the file name
    // (because of its positional parameter)
    files = argParser->getInputFileList();

    PluginManager::getInstance().init();

    //if (files.size() == 0)
    //{
    //    msg_error("") << "need >= 1 file";
    //    return 0;
    //}
    
    //// moved singletons
    // Task manager
    const bool DefaultTaskSchedulerRegistered = sofa::simulation::MainTaskSchedulerFactory::registerScheduler(
        sofa::simulation::DefaultTaskScheduler::name(),
        &sofa::simulation::DefaultTaskScheduler::create);



#ifdef WIN32
    std::string prefixScenes = "D:\\sofa\\src\\sandbox\\examples\\Demos\\" ;
#elif defined (__linux__)
    std::string prefixScenes = "/linuxdata/sofa/src/sandbox/examples/Demos/" ;
#else
    std::string prefixScenes = "/Users/fred/Work/sofa/src/master/examples/Demos/" ;
#endif
    files.resize(3);
    for(auto& file : files)
        file = prefixScenes + "caduceus_nomt.scn";

    std::vector<sofa::simulation::NodeSPtr> groots;
    for (const auto& filename : files)
    {
        auto groot = sofa::simulation::node::load(filename, false);
        if (!groot)
        {
            msg_error("") << "while loading " << filename;
            return 0;
        }

        // intersections
        //using namespace sofa::component::collision::detection::intersection;
        //sofa::core::collision::IntersectorCreator<MinProximityIntersection, MeshMinProximityIntersection> MeshMinProximityIntersectors("Mesh", groot->getContext());

        sofa::simulation::node::initRoot(groot.get());
        groot->setAnimate(true);

        groots.push_back(groot);
    }

    //if (defaultTaskScheduler)
    //{
    //    defaultTaskScheduler->addWorkerThread(0, std::string("SubMain "));
    //}
    // 
    // ownership in Main() i.e will start/stop
//    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
//    auto* defaultTaskScheduler = dynamic_cast<sofa::simulation::DefaultTaskScheduler*>(taskScheduler);
//    taskScheduler->init(0);


    auto simuLambda = [&](auto simuId)
        {
            std::size_t counter = 0;
            auto& groot = groots[simuId];

//            assert(defaultTaskScheduler);
//            auto* subMainThread = defaultTaskScheduler->addWorkerThread(0, std::string("SubMain "));

            while (counter < 1000)
            {
                msg_info("") << ">>>> " << simuId << " Step " << counter << " start ";
                sofa::simulation::node::animate(groot.get());
                msg_info("") << "<<<< " << simuId << " Step " << counter << " end ";
                counter++;
            }
//            subMainThread->setFinished();

        }
    ;

    // Run the main loop
    std::vector<std::thread> threads;

    for(std::size_t i = 0 ; i<groots.size() ; i++)
    {
        threads.emplace_back(simuLambda, i);
    }

    for(auto& t : threads)
    {
        t.join();
    }

//    {
//        std::jthread t0(simuLambda, 0);
//        std::jthread t1(simuLambda, 1);
//        std::jthread t2(simuLambda, 2);
//        std::jthread t3(simuLambda, 3);
//    }

    //std::for_each(std::execution::par_unseq, simuIds.begin(), simuIds.end(), simuLambda);


    //std::size_t counter = 0;
    //while (counter < 100)
    //{
    //    msg_info("") << ">> Step " << counter << " start";

    //    std::size_t simuId = 0;
    //    for (auto groot : groots)
    //    {

    //    }

    //    msg_info("") << "<< Step " << counter << " end";
    //    counter++;
    //}

    for (auto groot : groots)
    {
        if (groot != nullptr)
            sofa::simulation::node::unload(groot);
    }


    sofa::simulation::common::cleanup();
    sofa::simulation::graph::cleanup();
    return 0;
}
