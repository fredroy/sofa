cmake_minimum_required(VERSION 3.12)

project(sceneCreatorExamples)

sofa_find_package(SofaGui)
sofa_find_package(SceneCreator)

SET( HEADERS

    )

set( SOURCES
    SceneCreatorBenchmarks.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} SofaGui SceneCreator)
