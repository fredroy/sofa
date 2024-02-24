cmake_minimum_required(VERSION 3.12)
project(ParallelRunSofa)

sofa_find_package(Sofa.Simulation.Graph REQUIRED)
sofa_find_package(Sofa.GUI.Common REQUIRED)
sofa_find_package(Sofa.GUI.Batch REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)

if(NOT MSVC)
    find_package(TBB REQUIRED)
endif()

set(SOURCE_FILES
    Main.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Simulation.Graph)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Common)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Batch)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Collision.Detection.Intersection)
if(NOT MSVC)
    target_link_libraries(${PROJECT_NAME} tbb)
endif()
