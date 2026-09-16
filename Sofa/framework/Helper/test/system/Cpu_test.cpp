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
#include <gtest/gtest.h>
#include <sofa/helper/system/Cpu.h>

#include <thread>

namespace sofa
{

TEST(Cpu, logicalCoreCount)
{
    const unsigned logical = helper::system::getLogicalCoreCount();
    EXPECT_GE(logical, 1u);
    if (std::thread::hardware_concurrency() > 0)
    {
        EXPECT_EQ(logical, std::thread::hardware_concurrency());
    }
}

TEST(Cpu, physicalCoreCount)
{
    const unsigned physical = helper::system::getPhysicalCoreCount();
    const unsigned logical = helper::system::getLogicalCoreCount();

    EXPECT_GE(physical, 1u);
    EXPECT_LE(physical, logical);
}

TEST(Cpu, performanceCoreCount)
{
    const unsigned performance = helper::system::getPerformanceCoreCount();
    const unsigned physical = helper::system::getPhysicalCoreCount();

    EXPECT_GE(performance, 1u);
    EXPECT_LE(performance, physical);
}

} // namespace sofa
