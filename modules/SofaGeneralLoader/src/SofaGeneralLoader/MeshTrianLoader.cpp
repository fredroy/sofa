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
#include <sofa/core/ObjectFactory.h>
#include <SofaGeneralLoader/MeshTrianLoader.h>
#include <sofa/core/visual/VisualParams.h>

#include <iostream>
#include <fstream>

namespace sofa::component::loader
{

using namespace sofa::type;
using namespace sofa::defaulttype;

int MeshTrianLoaderClass = core::RegisterObject("Specific mesh loader for trian (only triangulations) file format.")
        .add< MeshTrianLoader >()
        ;

MeshTrianLoader::MeshTrianLoader() : MeshLoader()
    , p_trian2(initData(&p_trian2,(bool)false,"trian2","Set to true if the mesh is a trian2 format."))
    , neighborTable(initData(&neighborTable,"neighborTable","Table of neighborhood triangle indices for each triangle."))
    , edgesOnBorder(initData(&edgesOnBorder,"edgesOnBorder","List of edges which are on the border of the mesh loaded."))
    , trianglesOnBorderList(initData(&trianglesOnBorderList,"trianglesOnBorderList","List of triangle indices which are on the border of the mesh loaded."))
{
    neighborTable.setPersistent(false);
    edgesOnBorder.setPersistent(false);
    trianglesOnBorderList.setPersistent(false);
}



bool MeshTrianLoader::doLoad()
{
    msg_info() << "Loading Trian file: " << m_filename;

    bool fileRead = false;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();
    std::ifstream file(filename);

    if (!file.good())
    {
        msg_error() << "Cannot read file '" << m_filename << "'.";
        return false;
    }

    // -- Reading file
    if (p_trian2.getValue())
        fileRead = this->readTrian2 (filename);
    else
        fileRead = this->readTrian (filename);

    file.close();
    return fileRead;
}

void MeshTrianLoader::doClearBuffers()
{
    neighborTable.beginEdit()->clear();
    neighborTable.endEdit();
    edgesOnBorder.beginEdit()->clear();
    edgesOnBorder.endEdit();
    trianglesOnBorderList.beginEdit()->clear();
    trianglesOnBorderList.endEdit();
}



bool MeshTrianLoader::readTrian (const char* filename)
{
    std::ifstream dataFile (filename);



    // --- data used in trian files ---
    unsigned int nbVertices = 0;
    unsigned int nbTriangles = 0;
    sofa::helper::vector <unsigned int> the_edge;
    the_edge.resize (2);


    // --- Loading Vertices positions ---
    dataFile >> nbVertices; //Loading number of Vertex

    auto my_positions = getWriteOnlyAccessor(d_positions);
    for (unsigned int i=0; i<nbVertices; ++i)
    {
        SReal x,y,z;

        dataFile >> x >> y >> z;

        my_positions.push_back (Vector3(x, y, z));
    }

    // --- Loading Triangles array ---
    dataFile >> nbTriangles; //Loading number of Triangle

    auto my_triangles = getWriteOnlyAccessor(d_triangles);
    auto my_neighborTable = getWriteOnlyAccessor(neighborTable);
    auto my_edgesOnBorder = getWriteOnlyAccessor(edgesOnBorder);
    auto my_trianglesOnBorderList = getWriteOnlyAccessor(trianglesOnBorderList);

    for (unsigned int i=0; i<nbTriangles; ++i)
    {
        Triangle nodes;
        type::stdtype::fixed_array <int,3> ngh;

        dataFile >>  nodes[0] >> nodes[1] >> nodes[2] >> ngh[0] >> ngh[1] >> ngh[2];

        // set 3 triangle vertices  ==>> Dans le MeshLoader?

        addTriangle(my_triangles.wref(), nodes);
        my_neighborTable.push_back (ngh);

        // if we have a boundary edge store it in the m_edgeOnBorder array:
        if (ngh[0] == -1) // TODO : verifier la correspondance edgeArray et vectexIndices
        {
            the_edge[0] = nodes[1];
            the_edge[1] = nodes[2];

            my_edgesOnBorder.push_back (the_edge);
        }

        if (ngh[1] == -1) // TODO : verifier la correspondance edgeArray et vectexIndices
        {
            the_edge[0] = nodes[2];
            the_edge[1] = nodes[0];

            my_edgesOnBorder.push_back (the_edge);
        }

        if (ngh[2] == -1) // TODO : verifier la correspondance edgeArray et vectexIndices
        {
            the_edge[0] = nodes[0];
            the_edge[1] = nodes[1];

            my_edgesOnBorder.push_back (the_edge);
        }


        // if we have a boundary triangle store it (ONLY ONCE) in the m_triangleOnBorderList array:
        for (unsigned int j=0; j<3; ++j)
            if (ngh[j] == -1)
            {
                my_trianglesOnBorderList.push_back (i);
                break;
            }
    }

    return true;
}


bool MeshTrianLoader::readTrian2 (const char* filename)
{
    std::ifstream dataFile (filename);

    // --- data used in trian2 files ---
    unsigned int nbVertices = 0;
    unsigned int nbNormals = 0;
    unsigned int nbTriangles = 0;

    // --- TODO: Loading material parameters ---
    char buffer[256];
    for (unsigned int i=0; i<13; ++i)
    {
        dataFile.getline (buffer,256);
    }

    // --- Loading Vertices positions ---
    dataFile >> buffer >> nbVertices; //Loading number of Vertex

    auto my_positions = getWriteOnlyAccessor(d_positions);
    for (unsigned int i=0; i<nbVertices; ++i)
    {
        SReal x,y,z;

        dataFile >> x >> y >> z;
        my_positions.push_back (Vector3(x, y, z));
    }


    // --- Loading Normals positions ---
    dataFile >> buffer >> nbNormals; //Loading number of Vertex

    auto my_normals = getWriteOnlyAccessor(d_normals);
    for (unsigned int i=0; i<nbNormals; ++i)
    {
        SReal x,y,z;

        dataFile >> x >> y >> z;
        my_normals.push_back (Vector3(x, y, z));
    }


    // --- Loading Triangles array ---
    dataFile >> buffer >> nbTriangles; //Loading number of Triangle

    auto my_triangles = getWriteOnlyAccessor(d_triangles);

    for (unsigned int i=0; i<nbTriangles; ++i)
    {
        Triangle nodes;

        dataFile >>  nodes[0] >> nodes[1] >> nodes[2] ;

        addTriangle(my_triangles.wref(), nodes);
    }

    return true;
}



} //namespace sofa::component::loader
