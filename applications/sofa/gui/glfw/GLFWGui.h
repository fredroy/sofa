/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_SIMPLE_SIMPLEGLFW_H
#define SOFA_GUI_SIMPLE_SIMPLEGLFW_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <sofa/gui/BaseGUI.h>

#ifndef __APPLE__
#include <GL/glew.h>
#endif // __APPLE__


#include <sofa/gui/PickHandler.h>

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseVisual/InteractiveCamera.h>


#ifdef SOFA_BUILD_SOFAGUIGLFW
#    define SOFA_SOFAGUIGLFW_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#    define SOFA_SOFAGUIGLFW_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace sofa
{

namespace gui
{

namespace glfw
{

class SOFA_SOFAGUIGLFW_API GLFWGUI : public sofa::gui::BaseGUI
{

public:
    typedef sofa::core::visual::VisualParams VisualParams;

    /// @name methods each GUI must implement
    /// @{

    GLFWGUI();

    int mainLoop();
    void redraw();
    int closeGUI();

    sofa::simulation::Node* currentSimulation()
    {
        return getScene();
    }

    /// @}

    /// @name registration of each GUI
    /// @{

    static int InitGUI(const char* name, const std::vector<std::string>& options);
    static BaseGUI* CreateGUI(const char* name, const std::vector<std::string>& options, sofa::simulation::Node::SPtr groot = NULL, const char* filename = NULL);

    /// @}

protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    ~GLFWGUI();

public:

    // glut callbacks

    static GLFWGUI* instance;

    static void glfw_keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void glfw_reshape(GLFWwindow* window, int w, int h);
    static void glfw_mouse(GLFWwindow* window, int button, int action, int mods);
    static void glfw_motion(GLFWwindow* window, double x, double y);

private:

    enum
    {
        BTLEFT_MODE = 101,
        BTRIGHT_MODE = 102,
        BTMIDDLE_MODE = 103,
    };
    // Interaction
    enum
    {
        XY_TRANSLATION = 1,
        Z_TRANSLATION = 2,
    };

    enum { MINMOVE = 10 };


    sofa::simulation::Node::SPtr groot;
    std::string sceneFileName;
    sofa::component::visualmodel::BaseCamera::SPtr currentCamera;

    int                _W, _H;
    int                _clearBuffer;
    bool            _lightModelTwoSides;
    float            _lightPosition[4];
    int                _navigationMode;
    int                _mouseX, _mouseY;
    int                _savedMouseX, _savedMouseY;
    bool            _spinning;
    bool            _moving;
    bool            _video;
    bool            _axis;
    bool            _animationOBJ;
    int             _animationOBJcounter;
    int             _background;
    float            _zoomSpeed;
    float            _panSpeed;

    GLuint            _numOBJmodels;
    GLuint            _materialMode;
    GLboolean        _facetNormal;
    float            _zoom;
    int                _renderingMode;
    bool            _waitForRender;
    sofa::helper::system::thread::ctime_t _beginTime;
    double lastProjectionMatrix[16];
    double lastModelviewMatrix[16];
    GLint lastViewport[4];
    bool initTexturesDone;

public:

    void step();
    void animate();
    void playpause();
    void resetScene();
    void resetView();
    void saveView();

    void screenshot(int compression_level = -1);
    void exportOBJ(bool exportMTL=true);
    void dumpState(bool);

    void initializeGL();
    void paintGL();
    void resizeGL( int w, int h );

    void keyPressEvent ( int k );
    void keyReleaseEvent ( int k );

    enum EventType
    {
        MouseButtonPress, MouseMove, MouseButtonRelease
    };
    void mouseEvent ( int type, int x, int y, int bt );

    void eventNewStep();

protected:
    float getWindowPixelSizeRatio();
    void calcProjection();

public:
    void setScene(sofa::simulation::Node::SPtr scene, const char* filename=NULL, bool temporaryFile=false);
    sofa::simulation::Node* getScene()
    {
        return groot.get();
    }
    const std::string& getSceneFileName()
    {
        return sceneFileName;
    }
    void setCameraMode(core::visual::VisualParams::CameraType);
    void getView(sofa::defaulttype::Vec3d& pos, sofa::defaulttype::Quat& ori) const;
    void setView(const sofa::defaulttype::Vec3d& pos, const sofa::defaulttype::Quat &ori);
    void moveView(const sofa::defaulttype::Vec3d& pos, const sofa::defaulttype::Quat &ori);
    void newView();

    int getWidth()
    {
        return _W;
    };
    int getHeight()
    {
        return _H;
    };

    void updateOBJ();

    /////////////////
    // Interaction //
    /////////////////

    PickHandler pick;
    bool _mouseInteractorMoving;
    int _mouseInteractorSavedPosX;
    int _mouseInteractorSavedPosY;

private:

    void    drawAxis(double xpos, double ypos, double zpos, double arrowSize);
    void    drawBox(SReal* minBBox, SReal* maxBBox, double r=0.0);

    void    drawLogo();
    void    displayOBJs();
    void    displayMenu();
    void    drawScene();

protected:
    bool isControlPressed() const;
    bool isShiftPressed() const;
    bool isAltPressed() const;
    bool m_isControlPressed;
    bool m_isShiftPressed;
    bool m_isAltPressed;
    void updateModifiers(int modifiers);
    bool m_dumpState;
    bool m_displayComputationTime;
    std::ofstream* m_dumpStateStream;
    VisualParams* vparams;
    core::visual::DrawTool*   drawTool;
    GLFWwindow* m_window;
};

} // namespace glfw

} // namespace gui

} // namespace sofa

#endif // SOFA_GUI_SIMPLE_SIMPLEGLFW_H
