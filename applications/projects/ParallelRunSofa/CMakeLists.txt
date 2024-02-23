cmake_minimum_required(VERSION 3.12)
project(ParallelRunSofa)

sofa_find_package(Sofa.Simulation.Graph REQUIRED)
sofa_find_package(Sofa.GUI.Common REQUIRED)
sofa_find_package(Sofa.GUI.Batch REQUIRED)

set(SOURCE_FILES
    Main.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Simulation.Graph)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Common)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Batch)
