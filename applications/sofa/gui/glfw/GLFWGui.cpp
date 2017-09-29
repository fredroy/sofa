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
*                              SOFA :: GLFWGUI                                *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "GLFWGUI.h"

#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/io/ImageBMP.h>

#include <sofa/defaulttype/BoundingBox.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/gui/OperationFactory.h>
#include <sofa/gui/MouseOperations.h>

#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/helper/gl/Utilities.h>

namespace sofa
{

namespace gui
{

namespace glfw
{

using std::cout;
using std::endl;
using namespace sofa::defaulttype;
using namespace sofa::helper::gl;
using sofa::simulation::getSimulation;

GLFWGUI* GLFWGUI::instance = NULL;

int GLFWGUI::mainLoop()
{
    while(!glfwWindowShouldClose(m_window))
    {
        glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        if (this->getScene() && this->getScene()->getContext()->getAnimate())
            this->step();
        else
            sofa::helper::system::thread::CTime::sleep(0.01);

        this->animate();

        this->paintGL();

        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }

    return 0;
}

void GLFWGUI::redraw()
{
    this->paintGL();
}

int GLFWGUI::closeGUI()
{
    glfwDestroyWindow(m_window);
    glfwTerminate();
    return 0;
}


SOFA_DECL_CLASS(GLFWGUI)

static sofa::core::ObjectFactory::ClassEntry::SPtr classVisualModel;

int GLFWGUI::InitGUI(const char* /*name*/, const std::vector<std::string>& /*options*/)
{
    // Replace generic visual models
#ifdef SOFA_NO_OPENGL2
    sofa::core::ObjectFactory::AddAlias("VisualModel", "OglCoreModel", true,
            &classVisualModel);
    sofa::core::ObjectFactory::AddAlias("OglModel", "OglCoreModel", true,
            &classVisualModel);
#else
    sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true,
            &classVisualModel);
#endif // SOFA_NO_OPENGL2
    return 0;
}

BaseGUI* GLFWGUI::CreateGUI(const char* /*name*/, const std::vector<std::string>& /*options*/, sofa::simulation::Node::SPtr groot, const char* filename)
{
    if (!glfwInit())
        exit(EXIT_FAILURE);

    GLFWGUI* gui = new GLFWGUI();
    gui->setScene(groot, filename);

    gui->initializeGL();

    return gui;

}

void GLFWGUI::glfw_reshape(GLFWwindow* window, int w, int h)
{
    if (instance)
    {
        instance->resizeGL(w,h);
    }
}

void GLFWGUI::glfw_keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    if (instance)
    {
        instance->updateModifiers(mods);
        if(action == GLFW_PRESS)
            instance->keyPressEvent(key);
    }
}

void GLFWGUI::glfw_mouse(GLFWwindow* window, int button, int action, int mods)
{
    if (instance)
    {
        instance->updateModifiers(mods);
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        instance->mouseEvent( (action==GLFW_PRESS ? MouseButtonPress : MouseButtonRelease), x, y, button );
    }
}

void GLFWGUI::glfw_motion(GLFWwindow* window, double x, double y)
{
    if (instance)
    {
        instance->mouseEvent( MouseMove, x, y, 0 );
    }
}


// ---------------------------------------------------------
// --- Constructor
// ---------------------------------------------------------
GLFWGUI::GLFWGUI()
    : m_window(NULL)
{

    //TODO: uncomment this to force OpenGLCore context with MacOS
//#ifdef __APPLE__
//    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
//    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
//    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//#endif // __APPLE__

    m_window = glfwCreateWindow(640, 480, ":: SOFA ::", NULL, NULL);
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(0);

    instance = this;

    const unsigned char* glver = glGetString(GL_VERSION);
    std::cout << "OpenGL version: "  << glver << std::endl;

    // Here we initialize our multi-texturing functions
#if defined(SOFA_HAVE_GLEW) && !defined(PS3)
    glewInit();
    std::cout << "GLEW initialized" << std::endl;
#endif

    glfwSetKeyCallback(m_window, glfw_keyboard);
    glfwSetWindowSizeCallback(m_window, glfw_reshape);
    glfwSetCursorPosCallback(m_window, glfw_motion);
    glfwSetMouseButtonCallback(m_window, glfw_mouse);

    groot = NULL;
    initTexturesDone = false;
    // setup OpenGL mode for the window

    _zoom = 1.0;
    _zoomSpeed = 250.0;
    _panSpeed = 25.0;
    _navigationMode = 0;
    _spinning = false;
    _moving = false;
    _video = false;
    _animationOBJ = false;
    _axis = false;
    _background = 0;
    _numOBJmodels = 0;
    _materialMode = 0;
    _facetNormal = GL_FALSE;
    _renderingMode = GL_RENDER;
    _waitForRender = false;

    ////////////////
    // Interactor //
    ////////////////
    _mouseInteractorMoving = false;
    _mouseInteractorSavedPosX = 0;
    _mouseInteractorSavedPosY = 0;

    //////////////////////
    m_isControlPressed = false;
    m_isShiftPressed = false;;
    m_isAltPressed = false;
    m_dumpState = false;
    m_dumpStateStream = 0;

    //Register the different Operations possible
    RegisterOperation("Attach").add< AttachOperation >();
    RegisterOperation("Add recorded camera").add< AddRecordedCameraOperation >();
    RegisterOperation("Start navigation").add< StartNavigationOperation >();
    RegisterOperation("Fix").add< FixOperation >();
    RegisterOperation("Incise").add< InciseOperation >();
    RegisterOperation("Remove").add< TopologyOperation >();

    //Add to each button of the mouse an operation
    pick.changeOperation(LEFT,   "Attach");
    pick.changeOperation(MIDDLE, "Incise");
    pick.changeOperation(RIGHT,  "Remove");

    vparams = core::visual::VisualParams::defaultInstance();
    drawTool = new sofa::core::visual::DrawToolGL();
    vparams->drawTool() = drawTool;

}


// ---------------------------------------------------------
// --- Destructor
// ---------------------------------------------------------
GLFWGUI::~GLFWGUI()
{
    if (instance == this) instance = NULL;
}

// -----------------------------------------------------------------
// --- OpenGL initialization method - includes light definitions,
// --- color tracking, etc.
// -----------------------------------------------------------------
void GLFWGUI::initializeGL(void)
{
    static GLfloat    specref[4];
    static GLfloat    ambientLight[4];
    static GLfloat    diffuseLight[4];
    static GLfloat    specular[4];
    static GLfloat    lmodel_ambient[]    = {0.0f, 0.0f, 0.0f, 0.0f};
    static GLfloat    lmodel_twoside[]    = {GL_FALSE};
    static GLfloat    lmodel_local[]        = {GL_FALSE};
    static bool        initialized            = false;

    if (!initialized)
    {
        // Define light parameters
        //_lightPosition[0] = 0.0f;
        //_lightPosition[1] = 10.0f;
        //_lightPosition[2] = 0.0f;
        //_lightPosition[3] = 1.0f;

        _lightPosition[0] = -0.7f;
        _lightPosition[1] = 0.3f;
        _lightPosition[2] = 0.0f;
        _lightPosition[3] = 1.0f;

        ambientLight[0] = 0.5f;
        ambientLight[1] = 0.5f;
        ambientLight[2] = 0.5f;
        ambientLight[3] = 1.0f;

        diffuseLight[0] = 0.9f;
        diffuseLight[1] = 0.9f;
        diffuseLight[2] = 0.9f;
        diffuseLight[3] = 1.0f;

        specular[0] = 1.0f;
        specular[1] = 1.0f;
        specular[2] = 1.0f;
        specular[3] = 1.0f;

        specref[0] = 1.0f;
        specref[1] = 1.0f;
        specref[2] = 1.0f;
        specref[3] = 1.0f;

        _clearBuffer = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
        _lightModelTwoSides = false;

        glDepthFunc(GL_LEQUAL);
        glClearDepth(1.0);
        glEnable(GL_NORMALIZE);

        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

        // Set light model
        glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, lmodel_local);
        glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);

        // Setup 'light 0'
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
        glLightfv(GL_LIGHT0, GL_POSITION, _lightPosition);

        // Enable color tracking
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

        // All materials hereafter have full specular reflectivity with a high shine
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specref);
        glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 128);

        glShadeModel(GL_SMOOTH);

        // Define background color
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        //Load texture for logo
        //texLogo = new helper::gl::Texture(new helper::io::ImageBMP( sofa::helper::system::DataRepository.getFile("textures/SOFA_logo.bmp")));
        //texLogo->init();

#ifndef PS3
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
#endif
        // Turn on our light and enable color along with the light
        //glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        //glEnable(GL_COLOR_MATERIAL);

        // change status so we only do this stuff once
        initialized = true;

        _beginTime = sofa::helper::system::thread::CTime::getTime();

        printf("\n");
    }

    this->resizeGL(640,480);

    // switch to preset view
    resetView();
}

// ---------------------------------------------------
// ---
// ---
// ---------------------------------------------------
void GLFWGUI::drawAxis(double xpos, double ypos, double zpos,
        double arrowSize)
{
    float fontScale = (float) (arrowSize * 0.25);
/*
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING)ing;
    Enable<GL_COLOR_MATERIAL> colorMat;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(GL_SMOOTH);

    // --- Draw the "X" axis in red
    glPushMatrix();
    glColor3f(1.0, 0.0, 0.0);
    glTranslated(xpos, ypos, zpos);
    glRotatef(90.0f, 0.0, 1.0, 0.0);
    gluCylinder(_tube, arrowSize / 50.0, arrowSize / 50.0, arrowSize, 10, 10);
    glTranslated(0.0, 0.0, arrowSize);
    gluCylinder(_arrow, arrowSize / 15.0, 0.0, arrowSize / 5.0, 10, 10);
    // ---- Display a "X" near the tip of the arrow
    glTranslated(-0.5 * fontScale, arrowSize / 15.0, arrowSize / 5.0);

    helper::gl::GlText::draw('X', sofa::defaulttype::Vector3(0.0, 0.0, 0.0), fontScale);

    // --- Undo transforms
    glTranslated(-xpos, -ypos, -zpos);
    glPopMatrix();

    // --- Draw the "Y" axis in green
    glPushMatrix();
    glColor3f(0.0, 1.0, 0.0);
    glTranslated(xpos, ypos, zpos);
    glRotatef(-90.0f, 1.0, 0.0, 0.0);
    gluCylinder(_tube, arrowSize / 50.0, arrowSize / 50.0, arrowSize, 10, 10);
    glTranslated(0.0, 0.0, arrowSize);
    gluCylinder(_arrow, arrowSize / 15.0, 0.0, arrowSize / 5.0, 10, 10);
    // ---- Display a "Y" near the tip of the arrow
    glTranslated(-0.5 * fontScale, arrowSize / 15.0, arrowSize / 5.0);
    helper::gl::GlText::draw('Y', sofa::defaulttype::Vector3(0.0, 0.0, 0.0), fontScale);
    // --- Undo transforms
    glTranslated(-xpos, -ypos, -zpos);
    glPopMatrix();

    // --- Draw the "Z" axis in blue
    glPushMatrix();
    glColor3f(0.0, 0.0, 1.0);
    glTranslated(xpos, ypos, zpos);
    glRotatef(0.0f, 1.0, 0.0, 0.0);
    gluCylinder(_tube, arrowSize / 50.0, arrowSize / 50.0, arrowSize, 10, 10);
    glTranslated(0.0, 0.0, arrowSize);
    gluCylinder(_arrow, arrowSize / 15.0, 0.0, arrowSize / 5.0, 10, 10);
    // ---- Display a "Z" near the tip of the arrow
    glTranslated(-0.5 * fontScale, arrowSize / 15.0, arrowSize / 5.0);
    helper::gl::GlText::draw('Z', sofa::defaulttype::Vector3(0.0, 0.0, 0.0), fontScale);
    // --- Undo transforms
    glTranslated(-xpos, -ypos, -zpos);
    glPopMatrix();
*/
}

// ---------------------------------------------------
// ---
// ---
// ---------------------------------------------------
void GLFWGUI::drawBox(SReal* minBBox, SReal* maxBBox, double r)
{
 /*   //std::cout << "box = < " << minBBox[0] << ' ' << minBBox[1] << ' ' << minBBox[2] << " >-< " << maxBBox[0] << ' ' << maxBBox[1] << ' ' << maxBBox[2] << " >"<< std::endl;
    if (r==0.0)
        r = (Vector3(maxBBox) - Vector3(minBBox)).norm() / 500;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING)ing;
    Enable<GL_COLOR_MATERIAL> colorMat;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(GL_SMOOTH);

    // --- Draw the corners
    glColor3f(0.0, 1.0, 1.0);
    for (int corner=0; corner<8; ++corner)
    {
        glPushMatrix();
        glTranslated((corner&1)?minBBox[0]:maxBBox[0],
                (corner&2)?minBBox[1]:maxBBox[1],
                (corner&4)?minBBox[2]:maxBBox[2]);
        gluSphere(_sphere,2*r,20,10);
        glPopMatrix();
    }

    glColor3f(1.0, 1.0, 0.0);
    // --- Draw the X edges
    for (int corner=0; corner<4; ++corner)
    {
        glPushMatrix();
        glTranslated(           minBBox[0]           ,
                (corner&1)?minBBox[1]:maxBBox[1],
                (corner&2)?minBBox[2]:maxBBox[2]);
        glRotatef(90,0,1,0);
        gluCylinder(_tube, r, r, maxBBox[0] - minBBox[0], 10, 10);
        glPopMatrix();
    }

    // --- Draw the Y edges
    for (int corner=0; corner<4; ++corner)
    {
        glPushMatrix();
        glTranslated((corner&1)?minBBox[0]:maxBBox[0],
                minBBox[1]           ,
                (corner&2)?minBBox[2]:maxBBox[2]);
        glRotatef(-90,1,0,0);
        gluCylinder(_tube, r, r, maxBBox[1] - minBBox[1], 10, 10);
        glPopMatrix();
    }

    // --- Draw the Z edges
    for (int corner=0; corner<4; ++corner)
    {
        glPushMatrix();
        glTranslated((corner&1)?minBBox[0]:maxBBox[0],
                (corner&2)?minBBox[1]:maxBBox[1],
                minBBox[2]           );
        gluCylinder(_tube, r, r, maxBBox[2] - minBBox[2], 10, 10);
        glPopMatrix();
    }
    */
}


// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void GLFWGUI::drawLogo()
{
/*    int w = 0;
    int h = 0;

    if (texLogo && texLogo->getImage() && texLogo->getImage()->isLoaded())
    {
        h = texLogo->getImage()->getHeight();
        w = texLogo->getImage()->getWidth();
    }
    else return;

    Enable <GL_TEXTURE_2D> tex;
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-0.5, _W, -0.5, _H, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (texLogo)
        texLogo->bind();

    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d((_W-w)/2, (_H-h)/2, 0.0);

    glTexCoord2d(1.0, 0.0);
    glVertex3d( _W-(_W-w)/2, (_H-h)/2, 0.0);

    glTexCoord2d(1.0, 1.0);
    glVertex3d( _W-(_W-w)/2, _H-(_H-h)/2, 0.0);

    glTexCoord2d(0.0, 1.0);
    glVertex3d((_W-w)/2, _H-(_H-h)/2, 0.0);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    */
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void GLFWGUI::displayOBJs()
{

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    glShadeModel(GL_SMOOTH);
    //glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColor4f(1,1,1,1);
    glDisable(GL_COLOR_MATERIAL);


    vparams->sceneBBox() = groot->f_bbox.getValue();

    if (!initTexturesDone)
    {
//         std::cout << "-----------------------------------> initTexturesDone\n";
        //---------------------------------------------------
        simulation::getSimulation()->initTextures(groot.get());
        //---------------------------------------------------
        initTexturesDone = true;
    }

    {

        getSimulation()->draw(vparams,groot.get());

        if (_axis)
        {
            drawAxis(0.0, 0.0, 0.0, 10.0);
            if (vparams->sceneBBox().minBBox().x() < vparams->sceneBBox().maxBBox().x())
                drawBox(vparams->sceneBBox().minBBoxPtr(),
                        vparams->sceneBBox().maxBBoxPtr());
        }
    }

    // glDisable(GL_COLOR_MATERIAL);
}

// -------------------------------------------------------
// ---
// -------------------------------------------------------
void GLFWGUI::displayMenu(void)
{
    glDisable(GL_LIGHTING);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-0.5, _W, -0.5, _H, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(0.3f, 0.7f, 0.95f);
    glRasterPos2i(_W / 2 - 5, _H - 15);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void GLFWGUI::drawScene(void)
{
    if (!groot) return;
    if(!currentCamera)
    {
        std::cerr << "ERROR: no camera defined" << std::endl;
        return;
    }

    calcProjection();

    if (_background==0)
        drawLogo();

    glLoadIdentity();

    double mat[16];

    currentCamera->getOpenGLModelViewMatrix(mat);
    glMultMatrixd(mat);

    sofa::core::visual::VisualParams::defaultInstance()->setModelViewMatrix(mat);


    if (_renderingMode == GL_RENDER)
    {
        // Initialize lighting
        glPushMatrix();
        glLoadIdentity();
        glLightfv(GL_LIGHT0, GL_POSITION, _lightPosition);
        glPopMatrix();
        glEnable(GL_LIGHT0);

        glColor3f(0.5f, 0.5f, 0.6f);

        displayOBJs();

        displayMenu();        // always needs to be the last object being drawn
    }
}


// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void GLFWGUI::resizeGL(int width, int height)
{

    _W = width;
    _H = height;

    if(currentCamera)
        currentCamera->setViewport(width, height);

//    int wWidth, wHeight;
//    glfwGetWindowSize(m_window, &wWidth, &wHeight);

//    int fbWidth, fbHeight;
//    glfwGetFramebufferSize(m_window, &fbWidth, &fbHeight);

//    std::cout << wWidth << " x " << wHeight << std::endl;
//    std::cout << fbWidth << " x " << fbHeight << std::endl;

    calcProjection();
}

float GLFWGUI::getWindowPixelSizeRatio()
{
    int wWidth, wHeight;
    glfwGetWindowSize(m_window, &wWidth, &wHeight);

    int fbWidth, fbHeight;
    glfwGetFramebufferSize(m_window, &fbWidth, &fbHeight);


    if(wWidth < 1)
        return 1;

    return (float)fbWidth/(float)wWidth;
}

// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void GLFWGUI::calcProjection()
{
    int width = _W;
    int height = _H;
    double xNear, yNear/*, xOrtho, yOrtho*/;
    double xFactor = 1.0, yFactor = 1.0;
    double offset;
    double xForeground, yForeground, zForeground, xBackground, yBackground,
           zBackground;
    Vector3 center;

    /// Camera part
    if (!currentCamera)
        return;

    if (groot && (!groot->f_bbox.getValue().isValid() || _axis))
    {
        vparams->sceneBBox() = groot->f_bbox.getValue();
        currentCamera->setBoundingBox(vparams->sceneBBox().minBBox(), vparams->sceneBBox().maxBBox());
    }
    currentCamera->computeZ();

    vparams->zNear() = currentCamera->getZNear();
    vparams->zFar() = currentCamera->getZFar();
    ///

    xNear = 0.35 * vparams->zNear();
    yNear = 0.35 * vparams->zNear();
    offset = 0.001 * vparams->zNear(); // for foreground and background planes

    if ((height != 0) && (width != 0))
    {
        if (height > width)
        {
            xFactor = 1.0;
            yFactor = (double) height / (double) width;
        }
        else
        {
            xFactor = (double) width / (double) height;
            yFactor = 1.0;
        }
    }
    vparams->viewport() = sofa::helper::make_array(0,0,width,height);

    float pixelRatio = getWindowPixelSizeRatio();
    glViewport(0, 0, width*pixelRatio, height*pixelRatio);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    xFactor *= 0.01;
    yFactor *= 0.01;

    //std::cout << xNear << " " << yNear << std::endl;

    zForeground = -vparams->zNear() - offset;
    zBackground = -vparams->zFar() + offset;

    if (currentCamera->getCameraType() == core::visual::VisualParams::PERSPECTIVE_TYPE)
    {
        //gluPerspective(currentCamera->getFieldOfView(), (double) width / (double) height, vparams->zNear(), vparams->zFar());
        double projectionMatrix [16];

        float aspectRatio = float(width)/float(height);
        GLfloat znear = vparams->zNear();
        GLfloat zfar = vparams->zFar();
        float ymax, xmax;
        float temp, temp2, temp3, temp4;
        ymax = znear * tanf(currentCamera->getFieldOfView() * M_PI / 360.0);
        //ymin = -ymax;
        //xmin = -ymax * aspectRatio;
        xmax = ymax * aspectRatio;

        float left = -xmax;
        float right = xmax;
        float bottom = -ymax;
        float top = ymax;

        temp = 2.0 * znear;
        temp2 = right - left;
        temp3 = top - bottom;
        temp4 = zfar - znear;
        projectionMatrix[0] = temp / temp2;
        projectionMatrix[1] = 0.0;
        projectionMatrix[2] = 0.0;
        projectionMatrix[3] = 0.0;
        projectionMatrix[4] = 0.0;
        projectionMatrix[5] = temp / temp3;
        projectionMatrix[6] = 0.0;
        projectionMatrix[7] = 0.0;
        projectionMatrix[8] = (right + left) / temp2;
        projectionMatrix[9] = (top + bottom) / temp3;
        projectionMatrix[10] = (-zfar - znear) / temp4;
        projectionMatrix[11] = -1.0;
        projectionMatrix[12] = 0.0;
        projectionMatrix[13] = 0.0;
        projectionMatrix[14] = (-temp * zfar) / temp4;
        projectionMatrix[15] = 0.0;

        sofa::core::visual::VisualParams::defaultInstance()->setProjectionMatrix(&projectionMatrix[0]);

    }
    else
    {
        float ratio = (float)( vparams->zFar() / (vparams->zNear() * 20) );
        Vector3 tcenter = vparams->sceneTransform() * center;
        if (tcenter[2] < 0.0)
        {
            ratio = (float)( -300 * (tcenter.norm2()) / tcenter[2] );
        }
        glOrtho((-xNear * xFactor) * ratio, (xNear * xFactor) * ratio, (-yNear
                * yFactor) * ratio, (yNear * yFactor) * ratio,
                vparams->zNear(), vparams->zFar());
    }

    xForeground = -zForeground * xNear / vparams->zNear();
    yForeground = -zForeground * yNear / vparams->zNear();
    xBackground = -zBackground * xNear / vparams->zNear();
    yBackground = -zBackground * yNear / vparams->zNear();

    xForeground *= xFactor;
    yForeground *= yFactor;
    xBackground *= xFactor;
    yBackground *= yFactor;

    glMatrixMode(GL_MODELVIEW);
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void GLFWGUI::paintGL()
{

    // clear buffers (color and depth)
    if (_background==0)
        glClearColor(0.0f,0.0f,0.0f,0.0f);
    //glClearColor(0.0589f, 0.0589f, 0.0589f, 1.0f);
    else if (_background==1)
        glClearColor(0.0f,0.0f,0.0f,0.0f);
    else if (_background==2)
        glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClearDepth(1.0);
    glClear(_clearBuffer);

    // draw the scene
    drawScene();

    if (_video)
    {
        screenshot(2);
    }
}

void GLFWGUI::eventNewStep()
{
    static sofa::helper::system::thread::ctime_t beginTime[10];
    static const sofa::helper::system::thread::ctime_t timeTicks = sofa::helper::system::thread::CTime::getRefTicksPerSec();
    static int frameCounter = 0;
    if (frameCounter==0)
    {
        sofa::helper::system::thread::ctime_t t = sofa::helper::system::thread::CTime::getRefTime();
        for (int i=0; i<10; i++)
            beginTime[i] = t;
    }
    ++frameCounter;
    if ((frameCounter%10) == 0)
    {
        sofa::helper::system::thread::ctime_t curtime = sofa::helper::system::thread::CTime::getRefTime();
        int i = ((frameCounter/10)%10);
        double fps = ((double)timeTicks / (curtime - beginTime[i]))*(frameCounter<100?frameCounter:100);
        char buf[100];
        sprintf(buf, "%.1f FPS", fps);
        std::string title = "SOFA";
        if (!sceneFileName.empty())
        {
            title += " :: ";
            title += sceneFileName;
        }
        title += " :: ";
        title += buf;

        glfwSetWindowTitle(m_window, title.c_str());

        beginTime[i] = curtime;
        //frameCounter = 0;
    }
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void GLFWGUI::animate(void)
{
    if (_spinning)
    {
        //_newQuat = _currentQuat + _newQuat;
    }

    // update the entire scene
    redraw();
}


// ----------------------------------------
// --- Handle events (mouse, keyboard, ...)
// ----------------------------------------

bool GLFWGUI::isControlPressed() const
{
    return m_isControlPressed;
    //return glutGetModifiers()&GLUT_ACTIVE_CTRL;
}

bool GLFWGUI::isShiftPressed() const
{
    return m_isShiftPressed;
    //return glutGetModifiers()&GLUT_ACTIVE_SHIFT;
}

bool GLFWGUI::isAltPressed() const
{
    return m_isAltPressed;
    //return glutGetModifiers()&GLUT_ACTIVE_ALT;
}

void GLFWGUI::updateModifiers(int modifiers)
{
//#ifndef PS3
    m_isControlPressed =  (modifiers&GLFW_MOD_CONTROL )!=0;
    m_isShiftPressed   =  (modifiers&GLFW_MOD_SHIFT)!=0;
    m_isAltPressed     =  (modifiers&GLFW_MOD_ALT  )!=0;
//#endif
}

void GLFWGUI::keyPressEvent ( int k )
{
#ifndef PS3
    if( isControlPressed() ) // pass event to the scene data structure
    {
        //cerr<<"GLFWGUI::keyPressEvent, key = "<<k<<" with Control pressed "<<endl;
        sofa::core::objectmodel::KeypressedEvent keyEvent(k);
        groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
    }
    else  // control the GUI
        switch(k)
        {

        case 'S':
            // --- save screenshot
        {
            screenshot();
            break;
        }
        case 'V':
            // --- save video
        {
            _video = !_video;
            //capture.setCounter();
            break;
        }
        case 'W':
            // --- save current view
        {
            saveView();
            break;
        }

        case 'O':
            // --- export to OBJ
        {
            exportOBJ();
            break;
        }
        case 'P':
            // --- export to a succession of OBJ to make a video
        {
            _animationOBJ = !_animationOBJ;
            _animationOBJcounter = 0;
            break;
        }
        case  'R':
            // --- draw axis
        {
            _axis = !_axis;
            redraw();
            break;
        }
        case 'B':
            // --- change background
        {
            _background = (_background+1)%3;
            redraw();
            break;
        }

        case ' ':
            // --- start/stop
        {
            playpause();
            break;
        }

        case 'n':
            // --- step
        {
            step();
            redraw();
            break;
        }

        case 'q': //GLUT_KEY_Escape:
        {
            exit(0);
            break;
        }

        case GLFW_KEY_F5:
        {
            if (!sceneFileName.empty())
            {
                std::cout << "Reloading "<<sceneFileName<<std::endl;
                std::string filename = sceneFileName;
                Vec3d pos;
                Quat  ori;
                getView(pos, ori);

                simulation::Node::SPtr newroot = getSimulation()->load(filename.c_str());
                getSimulation()->init(newroot.get());
                if (newroot == NULL)
                {
                    std::cerr << "Failed to load "<<filename<<std::endl;
                    break;
                }
                setScene(newroot, filename.c_str());
                setView(pos, ori);
            }

          break;
        }
        }
#endif
}


void GLFWGUI::keyReleaseEvent ( int k )
{
    //cerr<<"GLFWGUI::keyReleaseEvent, key = "<<k<<endl;
    if( isControlPressed() ) // pass event to the scene data structure
    {
        sofa::core::objectmodel::KeyreleasedEvent keyEvent(k);
        groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
    }
}

// ---------------------- Here are the mouse controls for the scene  ----------------------
void GLFWGUI::mouseEvent ( int type, int eventX, int eventY, int button )
{
#ifndef PS3
    const sofa::core::visual::VisualParams::Viewport& viewport = vparams->viewport();

    MousePosition mousepos;
    mousepos.screenWidth  = viewport[2];
    mousepos.screenHeight = viewport[3];
    mousepos.x      = eventX;
    mousepos.y      = eventY;

    if( isShiftPressed() )
    {
        pick.activateRay(viewport[2],viewport[3], groot.get());
    }
    else
    {
        pick.deactivateRay();
    }

    if (isShiftPressed())
    {
        _moving = false;

        Vec3d p0, px, py, pz;
        double modelViewMatrix[16];
        double projectionMatrix[16];

        sofa::core::visual::VisualParams::defaultInstance()->getModelViewMatrix(&modelViewMatrix[0]);
        sofa::core::visual::VisualParams::defaultInstance()->getProjectionMatrix(&projectionMatrix[0]);

        sofa::helper::gl::Utilities::glhUnProject<double>(eventX, viewport[3]-1-(eventY), 0, modelViewMatrix, projectionMatrix, viewport.data(), &(p0[0]));
        sofa::helper::gl::Utilities::glhUnProject<double>(eventX+1, viewport[3]-1-(eventY), 0, modelViewMatrix, projectionMatrix, viewport.data(), &(px[0]));
        sofa::helper::gl::Utilities::glhUnProject<double>(eventX, viewport[3]-1-(eventY+1), 0, modelViewMatrix, projectionMatrix, viewport.data(), &(py[0]));
        sofa::helper::gl::Utilities::glhUnProject<double>(eventX, viewport[3]-1-(eventY), 1, modelViewMatrix, projectionMatrix, viewport.data(), &(pz[0]));

        px -= p0;
        py -= p0;
        pz -= p0;
        px.normalize();
        py.normalize();
        pz.normalize();
        Mat4x4d transform;
        transform.identity();
        transform[0][0] = px[0];
        transform[1][0] = px[1];
        transform[2][0] = px[2];
        transform[0][1] = py[0];
        transform[1][1] = py[1];
        transform[2][1] = py[2];
        transform[0][2] = pz[0];
        transform[1][2] = pz[1];
        transform[2][2] = pz[2];
        transform[0][3] = p0[0];
        transform[1][3] = p0[1];
        transform[2][3] = p0[2];
        Mat3x3d mat; mat = transform;
        Quat q; q.fromMatrix(mat);

        Vec3d position, direction;
        position  = transform*Vec4d(0,0,0,1);
        direction = transform*Vec4d(0,0,1,0);
        direction.normalize();
        pick.updateRay(position, direction);
        pick.updateMouse2D(mousepos);
        switch (type)
        {
            case MouseButtonPress:
                if (button == GLFW_MOUSE_BUTTON_LEFT) // Shift+Leftclick to deform the mesh
                {
                    pick.handleMouseEvent(PRESSED, LEFT);
                }
                else if (button == GLFW_MOUSE_BUTTON_RIGHT) // Shift+Rightclick to remove triangles
                {
                    pick.handleMouseEvent(PRESSED, RIGHT);
                }
                else if (button == GLFW_MOUSE_BUTTON_MIDDLE) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
                {
                    pick.handleMouseEvent(PRESSED, MIDDLE);
                }
                break;
            case MouseButtonRelease:
                //if (button == GLFW_MOUSE_BUTTON_LEFT)
                    {
                    if (button == GLFW_MOUSE_BUTTON_LEFT) // Shift+Leftclick to deform the mesh
                    {
                        pick.handleMouseEvent(RELEASED, LEFT);
                    }
                    else if (button == GLFW_MOUSE_BUTTON_RIGHT) // Shift+Rightclick to remove triangles
                    {
                        pick.handleMouseEvent(RELEASED, RIGHT);
                    }
                    else if (button == GLFW_MOUSE_BUTTON_MIDDLE) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
                    {
                        pick.handleMouseEvent(RELEASED, MIDDLE);
                    }
            }
            break;
            default: break;
        }
    }
    else if (isAltPressed())
    {
        _moving = false;

        _mouseInteractorSavedPosX = eventX;
        _mouseInteractorSavedPosY = eventY;

        switch (type)
        {
            case MouseButtonPress:
                // Mouse left button is pushed
                if (button == GLFW_MOUSE_BUTTON_LEFT)
                {
                    _navigationMode = BTLEFT_MODE;
                    _mouseInteractorMoving = true;
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
                // Mouse right button is pushed
                else if (button == GLFW_MOUSE_BUTTON_RIGHT)
                {
                    _navigationMode = BTRIGHT_MODE;
                    _mouseInteractorMoving = true;
                }
                // Mouse middle button is pushed
                else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
                {
                    _navigationMode = BTMIDDLE_MODE;
                    _mouseInteractorMoving = true;
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
                break;

            case MouseMove:
                //
                break;

            case MouseButtonRelease:
                // Mouse left button is released
                if (_mouseInteractorMoving)
                    _mouseInteractorMoving = false;

                if (button == GLFW_MOUSE_BUTTON_LEFT)
                {
                    ;//
                }
                // Mouse right button is released
                else if (button == GLFW_MOUSE_BUTTON_RIGHT)
                {
                    ;//
                }
                // Mouse middle button is released
                else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
                {
                    ;//
                }
                break;

            default:
                break;
        }

        if(_mouseInteractorMoving)
        {
            int dx = eventX - _mouseInteractorSavedPosX;
            int dy = eventY - _mouseInteractorSavedPosY;

            switch(_navigationMode)
            {
                case BTLEFT_MODE:
                    //TODO: moving light ?
                break;
                case BTMIDDLE_MODE:
                break;
                case BTRIGHT_MODE:
                break;
            }
            _mouseInteractorSavedPosX = eventX;
            _mouseInteractorSavedPosY = eventY;
        }
    }
    else if (isControlPressed())
    {
        //Control press stuff
    }
    else
    {
        switch (type)
        {
        case MouseButtonPress:
        {
            sofa::core::objectmodel::MouseEvent* mEvent = NULL;
            if (button == GLFW_MOUSE_BUTTON_LEFT)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed, eventX, eventY);
            else if (button == GLFW_MOUSE_BUTTON_RIGHT)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed, eventX, eventY);
            else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed, eventX, eventY);
            else{
                // A fallback event to rules them all...
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::AnyExtraButtonPressed, eventX, eventY);
            }
            currentCamera->manageEvent(mEvent);
            _moving = true;
            _spinning = false;
            _mouseX = eventX;
            _mouseY = eventY;
            break;
        }
        case MouseMove:
        {
            sofa::core::objectmodel::MouseEvent me(sofa::core::objectmodel::MouseEvent::Move, eventX, eventY);
            currentCamera->manageEvent(&me);
            break;
        }

        case MouseButtonRelease:
        {
            sofa::core::objectmodel::MouseEvent* mEvent = NULL;
            if (button == GLFW_MOUSE_BUTTON_LEFT)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased, eventX, eventY);
            else if (button == GLFW_MOUSE_BUTTON_RIGHT)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased, eventX, eventY);
            else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased, eventX, eventY);
            else{
                // A fallback event to rule them all...
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::AnyExtraButtonReleased, eventX, eventY);
            }
            currentCamera->manageEvent(mEvent);
            _moving = false;
            _spinning = false;
            _mouseX = eventX;
            _mouseY = eventY;
            break;
        }

        default:
            break;
        }

        redraw();
    }

#endif // PS3

}

void GLFWGUI::step()
{
    getSimulation()->animate(groot.get());
    getSimulation()->updateVisual(groot.get());

    if( m_dumpState )
        getSimulation()->dumpState( groot.get(), *m_dumpStateStream );

    eventNewStep();

    redraw();

    if (_animationOBJ)
    {
        exportOBJ(false);
        ++_animationOBJcounter;
    }
}

void GLFWGUI::playpause()
{
    if (groot)
    {
        groot->getContext()->setAnimate(!groot->getContext()->getAnimate());
    }
}

void GLFWGUI::dumpState(bool value)
{
    m_dumpState = value;
    if( m_dumpState )
    {
        m_dumpStateStream = new std::ofstream("dumpState.data");
    }
    else if( m_dumpStateStream!=NULL )
    {
        delete m_dumpStateStream;
        m_dumpStateStream = 0;
    }
}

void GLFWGUI::resetScene()
{
    if (groot)
    {
        getSimulation()->reset(groot.get());
        redraw();
    }
}

void GLFWGUI::resetView()
{
    bool fileRead = false;

    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName + ".view";
        fileRead = currentCamera->importParametersFromFile(viewFileName);
    }

    //if there is no .view file , look at the center of the scene bounding box
    // and with a Up vector in the same axis as the gravity
    if (!fileRead)
    {
        newView();
    }
    redraw();
}

void GLFWGUI::setCameraMode(core::visual::VisualParams::CameraType mode)
{
    currentCamera->setCameraType(mode);
}

void GLFWGUI::getView(Vec3d& pos, Quat& ori) const
{
    if (!currentCamera)
        return;

    const Vec3d& camPosition = currentCamera->getPosition();
    const Quat& camOrientation = currentCamera->getOrientation();

    pos[0] = camPosition[0];
    pos[1] = camPosition[1];
    pos[2] = camPosition[2];

    ori[0] = camOrientation[0];
    ori[1] = camOrientation[1];
    ori[2] = camOrientation[2];
    ori[3] = camOrientation[3];
}

void GLFWGUI::setView(const Vec3d& pos, const Quat &ori)
{
    Vec3d position;
    Quat orientation;
    for (unsigned int i=0 ; i<3 ; i++)
    {
        position[i] = pos[i];
        orientation[i] = ori[i];
    }
    orientation[3] = ori[3];

    if (currentCamera)
        currentCamera->setView(position, orientation);

    redraw();
}

void GLFWGUI::moveView(const Vec3d& pos, const Quat &ori)
{
    if (!currentCamera)
        return;

    currentCamera->moveCamera(pos, ori);
    redraw();
}

void GLFWGUI::newView()
{
    if (!currentCamera || !groot)
        return;

    currentCamera->setDefaultView(groot->getGravity());
}

void GLFWGUI::saveView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName + ".view";

        if(currentCamera->exportParametersInFile(viewFileName))
            std::cout << "View parameters saved in " << viewFileName << std::endl;
        else
            std::cout << "Error while saving view parameters in " << viewFileName << std::endl;
    }
}

void GLFWGUI::screenshot(int compression_level)
{
    //capture.saveScreen(compression_level);
}

void GLFWGUI::exportOBJ(bool exportMTL)
{
    if (!groot) return;
    std::ostringstream ofilename;
    if (!sceneFileName.empty())
    {
        const char* begin = sceneFileName.c_str();
        const char* end = strrchr(begin,'.');
        if (!end) end = begin + sceneFileName.length();
        ofilename << std::string(begin, end);
    }
    else
        ofilename << "scene";

    std::stringstream oss;
    oss.width(5);
    oss.fill('0');
    oss << _animationOBJcounter;

    ofilename << '_' << (oss.str().c_str());
    ofilename << ".obj";
    std::string filename = ofilename.str();
    std::cout << "Exporting OBJ Scene "<<filename<<std::endl;
    getSimulation()->exportOBJ(groot.get(), filename.c_str(),exportMTL);
}

void GLFWGUI::setScene(sofa::simulation::Node::SPtr scene, const char* filename, bool)
{
    std::ostringstream ofilename;

    sceneFileName = (filename==NULL)?"":filename;
    if (!sceneFileName.empty())
    {
        const char* begin = sceneFileName.c_str();
        const char* end = strrchr(begin,'.');
        if (!end) end = begin + sceneFileName.length();
        ofilename << std::string(begin, end);
        ofilename << "_";
    }
    else
        ofilename << "scene_";

    //capture.setPrefix(ofilename.str());
    groot = scene;
    initTexturesDone = false;

    //Camera initialization
    if (groot)
    {
        groot->get(currentCamera);
        if (!currentCamera)
        {
            currentCamera = sofa::core::objectmodel::New<component::visualmodel::InteractiveCamera>();
            currentCamera->setName(core::objectmodel::Base::shortName(currentCamera.get()));
            groot->addObject(currentCamera);
            currentCamera->p_position.forceSet();
            currentCamera->p_orientation.forceSet();
            currentCamera->bwdInit();
            resetView();
        }

        vparams->sceneBBox() = groot->f_bbox.getValue();
        currentCamera->setBoundingBox(vparams->sceneBBox().minBBox(), vparams->sceneBBox().maxBBox());

        // init pickHandler
        pick.init(groot.get());
    }
    redraw();
}

} // namespace glfw

} // namespace gui

} // namespace sofa
