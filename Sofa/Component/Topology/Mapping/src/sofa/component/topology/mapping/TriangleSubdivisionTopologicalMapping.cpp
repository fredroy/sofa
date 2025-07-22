/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/topology/mapping/TriangleSubdivisionTopologicalMapping.h>


#include <sofa/type/Vec.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>
#include <sofa/core/topology/TopologyChange.h>

#include <map>

namespace sofa::component::topology::mapping
{

using namespace sofa::defaulttype;
using namespace sofa::component::topology::container::dynamic;
using namespace sofa::core::topology;
using namespace sofa::core::behavior;

void registerTriangleSubdivisionTopologicalMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Special case of mapping where TriangleSetTopology is converted into a finer TriangleSetTopology.")
        .add< TriangleSubdivisionTopologicalMapping >());
}

// Implementation
TriangleSubdivisionTopologicalMapping::TriangleSubdivisionTopologicalMapping ()
   : d_inputPositions( initData(&d_inputPositions,"inputPositions","vertices coordinates of the upper topology.") )
   , d_inputTriangles( initData(&d_inputTriangles,"inputTriangles","triangle array of the upper topology.") )
   , d_outputPositions( initData(&d_outputPositions,"outputPositions","vertices coordinates of the lower topology.") )
   , d_outputTriangles( initData(&d_outputTriangles,"outputTriangles","triangle array of the lower topology.") )
   , d_subdivisions(initData(&d_subdivisions, 3, "subdivisions", "Number of subdivisions"))
{
}

void TriangleSubdivisionTopologicalMapping::init()
{
   //std::cout << "TriangleSubdivisionTopologicalMapping::init()" << std::endl;
    // Writes subdivided vertices and triangles into new topology

    if(!fromModel)
    {
        msg_error() << "TriangleSubdivisionTopologicalMapping requires an input triangular topology";
        return;
    }


    // Retrieves original topology
    TriangleSetTopologyContainer *from_tstc;
    fromModel->getContext()->get(from_tstc);

    if(!from_tstc)
    {
        msg_error() << "TriangleSubdivisionTopologicalMapping can't find input triangular topology";
        return;
    }


    SeqTriangles triangles;

    // List of in triangles
    if (d_inputTriangles.isSet())
    {
       triangles = d_inputTriangles.getValue();
    }
    else
    {
       triangles = from_tstc->getTriangles();
       // add Data link here!
       //m_inputTriangles.setParent(from_tstc->getTriangleDataArray());
    }

    if (triangles.size() == 0)
    {
        msg_error() << "TriangleSubdivisionTopologicalMapping requires an input triangular topology";
        return;
    }


    MechanicalState<Vec3Types>* mStateIn = dynamic_cast<MechanicalState<Vec3Types>*> (fromModel->getContext()->getMechanicalState());
    if (!mStateIn)
    {
        msg_error() << "TriangleSubdivisionTopologicalMapping can't access MechanicalState";
        return;
    }


    //const VecCoord &inRestVertices = *mStateIn->getX0();
    //const VecCoord &inVertices = *mStateIn->getX();
/*
    if (!m_inputPositions.isSet())
    {
       // add Data link here!
     //  m_inputPositions.setParent(mStateIn->State)
    }
*/

    // Initialises the list of vertices at rest and current positions
    this->subdivideVertices();


    // Iterates over the triangles to subdivide them at initial position (topology is created here)
    this->subdivideTriangles();

/* TODO: check if still needed
    if(toModel)
    {
       TriangleSetTopologyContainer *to_tstc;
       toModel->getContext()->get(to_tstc);
       to_tstc->clear();
       to_tstc->setNbPoints( subVertices.size() );

       for (unsigned int i=0; i<subVertices.size(); i++)
       {
          to_tstc->addPoint(subVertices[i][0], subVertices[i][1], subVertices[i][2]);
       }

       for (unsigned int i=0; i<subTriangles.size(); i++)
       {
          to_tstc->addTriangle(subTriangles[i][0], subTriangles[i][1], subTriangles[i][2]);
       }

       // Add Data link here?
    }
    */

    addInput(&d_inputPositions);
    addInput(&d_inputTriangles);

    addOutput(&d_outputPositions);
    addOutput(&d_outputTriangles);
    setDirtyValue();
    //std::cout << "TriangleSubdivisionTopologicalMapping::init() end" << std::endl;
}


void TriangleSubdivisionTopologicalMapping::subdivideVertices()
{
    //std::cout << "TriangleSubdivisionTopologicalMapping::subdivideVertices()" << std::endl;
    TriangleSetTopologyContainer *from_tstc;
    fromModel->getContext()->get(from_tstc);
    if (!from_tstc)
    {
        msg_error() << "TriangleSubdivisionTopologicalMapping can't create subverticies, upper topology not found.";
        return;
    }

    // TODO check if need to create edge output Data.
    // Iterates over edges to subdivide them
    SeqEdges subEdgesTemp;
    // Retrieves list of in edges
    const SeqEdges edges = from_tstc->getEdges();
    const VecCoord& positions = d_inputPositions.getValue();
    VecCoord& subVertices = *d_outputPositions.beginEdit();

    subVertices.clear();
    subVertices = positions;

    m_innerTriangles.clear();
    m_innerTriangles.resize(from_tstc->getNbTriangles());

    unsigned int cpt_id_position = subVertices.size()-1;

    for (unsigned int i=0; i<edges.size(); ++i)
    {
       Edge the_edge = edges[i];
       Coord pointA = positions[ the_edge[0] ];
       Coord pointB = positions[ the_edge[1] ];

       Coord pointC = (pointA + pointB)/2;
       subVertices.push_back(pointC);
       cpt_id_position++;

       // update inner triangle list
       sofa::type::vector<unsigned int> triAroundEdge = from_tstc->getTrianglesAroundEdge(i);
       for(unsigned int j=0; j<triAroundEdge.size(); ++j)
       {
          unsigned int triID = triAroundEdge[j];
          int EdgeIDinTri = from_tstc->getEdgeIndexInTriangle(from_tstc->getEdgesInTriangle(triID), i);

          if (EdgeIDinTri == -1)
          {
              msg_error() << "edge not found in triangle.";
              continue;
          }

          m_innerTriangles[triID][(EdgeIDinTri+1)%3] = cpt_id_position;
       }
    }


    d_outputPositions.endEdit();
    //std::cout << "TriangleSubdivisionTopologicalMapping::subdivideVertices() end" << std::endl;
    return;
}



void TriangleSubdivisionTopologicalMapping::subdivideTriangles()
{
   //std::cout << "TriangleSubdivisionTopologicalMapping::subdivideTriangles()" << std::endl;
   TriangleSetTopologyContainer *from_tstc;
   fromModel->getContext()->get(from_tstc);
   if (!from_tstc)
   {
       msg_error() << "can't create subTriangles, upper topology not found.";
       return;
   }

    sofa::helper::ReadAccessor< Data< SeqTriangles > > inputTriangles = d_inputTriangles;
    sofa::helper::WriteAccessor< Data< SeqTriangles > > outputTriangles = d_outputTriangles;

   outputTriangles.clear();

   if (inputTriangles.size() != m_innerTriangles.size())
   {
       msg_error() << "TriangleSubdivisionTopologicalMapping can't create subTriangles, inner triangles list is not of the good size.";
       return;
   }

   // for each triangle create the 4 inner triangles
   for (unsigned int i=0; i<inputTriangles.size(); ++i)
   {
      Triangle uperTri = inputTriangles[i];
      Triangle innerTri = m_innerTriangles[i];
      unsigned int ptA = uperTri[0];
      unsigned int ptB = uperTri[1];
      unsigned int ptC = uperTri[2];

      unsigned int ptAp = innerTri[0];
      unsigned int ptBp = innerTri[1];
      unsigned int ptCp = innerTri[2];

      outputTriangles.push_back(Triangle(ptA, ptAp, ptCp));
      outputTriangles.push_back(Triangle(ptAp, ptB, ptBp));
      outputTriangles.push_back(Triangle(ptCp, ptBp, ptC));
      outputTriangles.push_back(Triangle(ptAp, ptBp, ptCp));
   }


   //std::cout << "TriangleSubdivisionTopologicalMapping::subdivideTriangles() end" << std::endl;
   return;
}


void TriangleSubdivisionTopologicalMapping::propagateInformation()
{

}



void TriangleSubdivisionTopologicalMapping::doUpdate()
{
   //std::cout << "TriangleSubdivisionTopologicalMapping::update() tried implementation!" << std::endl;
   //if (m_inputPositions.isDirty())
      this->subdivideVertices();

   if (d_inputTriangles.isDirty())
      this->subdivideTriangles();

   cleanDirty();
   //std::cout << "TriangleSubdivisionTopologicalMapping::update() end." << std::endl;
   return;
}

/*old code for rigid:
  TODO: see if it is still used and needed !

    // Initialises the list of vertices at rest and current positions
    subRestVertices.clear();
    subVertices.clear();
    for (unsigned int i=0; i<inRestVertices.size(); i++)
    {
       subRestVertices.push_back(inRestVertices[i].getCenter());
       subVertices.push_back(inVertices[i].getCenter());
    }

    std::cout << "la 4" << std::endl;

    // Iterates over the rest triangles to subdivide them at rest position (just adds vertices)
    for (unsigned int t=0; t<triangles.size(); t++)
    {
       // Adds base triangle
       subTrianglesTemp.clear();
       subTrianglesTemp.push_back( triangles[t] );

       // Recursively divides the triangle into smaller ones
       for (int sub=0; sub<subdivisions.getValue(); sub++)
       {
          subdivide(subRestVertices, subTrianglesTemp, newSubTriangles);
          subTrianglesTemp = newSubTriangles;
          newSubTriangles.clear();
       }
    }

    std::cout << "la 5" << std::endl;

    // Iterates over the triangles to subdivide them at initial position (topology is created here)
    for (unsigned int t=0; t<triangles.size(); t++)
    {
       // Adds base triangle
       subTrianglesTemp.clear();
       subTrianglesTemp.push_back( triangles[t] );

       // Recursively divides the triangle into smaller ones
       for (int sub=0; sub<subdivisions.getValue(); sub++)
       {
          subdivide(subVertices, subTrianglesTemp, newSubTriangles);
          subTrianglesTemp = newSubTriangles;
          newSubTriangles.clear();
       }

       // Adds the subtriangles to the global list of subdivided triangles
       for (unsigned int i=0; i<subTrianglesTemp.size(); i++)
       {
          subTriangles.push_back(subTrianglesTemp[i]);
       }
    }

    std::cout << "la 6" << std::endl;
    // Iterates over edges to subdivide them
    SeqEdges subEdgesTemp, newSubEdges;
    // Retrieves list of in edges
    const SeqEdges edges = from_tstc->getEdges();
    for (unsigned int e=0; e<edges.size(); e++)
    {
       subEdgesTemp.clear();
       subEdgesTemp.push_back( edges[e] );

       // Recursively divides the edge into smaller ones
       for (int sub=0; sub<subdivisions.getValue(); sub++)
       {
          subdivide(subVertices, subEdgesTemp, newSubEdges);
          subEdgesTemp = newSubEdges;
          newSubEdges.clear();
       }

       // Adds the subedges to the global list of subdivided edges
       for (unsigned int i=0; i<subEdgesTemp.size(); i++)
       {
          subEdges.push_back(subEdgesTemp[i]);
       }
    }
    std::cout << "la 7" << std::endl;

    if(toModel)
    {
       TriangleSetTopologyContainer *to_tstc;
       toModel->getContext()->get(to_tstc);
       to_tstc->clear();
       to_tstc->setNbPoints( subVertices.size() );

       for (unsigned int i=0; i<subVertices.size(); i++)
       {
          to_tstc->addPoint(subVertices[i][0], subVertices[i][1], subVertices[i][2]);
       }

       for (unsigned int i=0; i<subTriangles.size(); i++)
       {
          to_tstc->addTriangle(subTriangles[i][0], subTriangles[i][1], subTriangles[i][2]);
       }

       // Add Data link here?

    }
    */




// --------------------------------------------------------------------------------------
// Subdivides each triangle into 4 by taking the middle of each edge
// --------------------------------------------------------------------------------------
void TriangleSubdivisionTopologicalMapping::subdivide(InitTypes::VecCoord &subVertices, const SeqTriangles subTriangles, SeqTriangles &newSubTriangles)
{
    for (unsigned int i=0; i<subTriangles.size(); i++)
    {
        Coord a, b, c;
        a = subVertices[(int)subTriangles[i][0]];
        b = subVertices[(int)subTriangles[i][1]];
        c = subVertices[(int)subTriangles[i][2]];

        // Global coordinates
        Coord mAB, mAC, mBC;
        mAB = (a+b)/2;
        mAC = (a+c)/2;
        mBC = (b+c)/2;

        // Adds vertex if we deal with a new point
        int indexAB, indexAC, indexBC;
        addVertexAndFindIndex(subVertices, mAB, indexAB);
        addVertexAndFindIndex(subVertices, mAC, indexAC);
        addVertexAndFindIndex(subVertices, mBC, indexBC);

        // Finds index of the 3 vertices
        int indexA, indexB, indexC;
        addVertexAndFindIndex(subVertices, a, indexA);
        addVertexAndFindIndex(subVertices, b, indexB);
        addVertexAndFindIndex(subVertices, c, indexC);

        // Adds the 4 subdivided triangles to the list
        newSubTriangles.push_back(Triangle(indexA, indexAB, indexAC));
        newSubTriangles.push_back(Triangle(indexAB, indexB, indexBC));
        newSubTriangles.push_back(Triangle(indexAC, indexBC, indexC));
        newSubTriangles.push_back(Triangle(indexBC, indexAC, indexAB));
    }
}

// --------------------------------------------------------------------------------------
// Subdivides each edge into 2 smaller edges
// --------------------------------------------------------------------------------------
void TriangleSubdivisionTopologicalMapping::subdivide(InitTypes::VecCoord &subVertices, const SeqEdges subEdges, SeqEdges &newSubEdges)
{
    for (unsigned int i=0; i<subEdges.size(); i++)
    {
        Coord a, b;
        a = subVertices[(int)subEdges[i][0]];
        b = subVertices[(int)subEdges[i][1]];

        // Global coordinates
        Coord mAB;
        mAB = (a+b)/2;

        // Finds index each of the 3 vertices
        int indexA, indexB, indexAB;
        addVertexAndFindIndex(subVertices, a, indexA);
        addVertexAndFindIndex(subVertices, b, indexB);
        addVertexAndFindIndex(subVertices, mAB, indexAB);

        // Adds edges if correct level of subdivision
        newSubEdges.push_back(Edge(indexA, indexAB));
        newSubEdges.push_back(Edge(indexAB, indexB));
    }
}


// --------------------------------------------------------------------------------------
// Adds a vertex if it is not already in the list
// --------------------------------------------------------------------------------------
void TriangleSubdivisionTopologicalMapping::addVertexAndFindIndex(InitTypes::VecCoord &subVertices, const Coord &vertex, int &index)
{
    bool alreadyHere = false;

    for (unsigned v=0; v<subVertices.size(); v++)
    {
        if ( (subVertices[v]-vertex).norm() < 0.0000001)
        {
            alreadyHere = true;
            index = v;
        }
    }
    if (alreadyHere == false)
    {
        subVertices.push_back(vertex);
        index = (int)subVertices.size()-1;
    }
}



void TriangleSubdivisionTopologicalMapping::updateTopologicalMappingTopDown()
{

}

void TriangleSubdivisionTopologicalMapping::updateTopologicalMappingBottomUp()
{

}

} // namespace sofa::component::topology::mapping
