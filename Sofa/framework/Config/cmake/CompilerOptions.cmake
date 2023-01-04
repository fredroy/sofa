#### Compiler options
# Set default compiler options (as defined in toolchain files)
# https://stackoverflow.com/questions/45995784/how-to-set-compiler-options-with-cmake-in-visual-studio-2017

if (MSVC)
    # remove default exceptions flags handling from CMAKE_CXX_FLAGS_INIT (which will initialize CMAKE_CXX_FLAGS)
    # it allows us to freely define them later
    set(CMAKE_CXX_FLAGS_INIT "/DWIN32 /D_WINDOWS /W3 /GR")
endif()
