cmake_minimum_required(VERSION 3.12)

find_package(Sofa.Config)

sofa_add_subdirectory(plugin GL SofaCUDA.GL)


if(TARGET SofaCUDA AND NOT TARGET SofaCUDA.GL)
    message(WARNING "SofaCUDA.GL is not active.")
    message("OpenGL-related code won't be enabled (Sharing OpenGL buffers with CUDA)")
endif()
