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
#pragma once

#include <sofa/helper/config.h>

namespace sofa::helper::system
{

/// @brief Number of logical processors (hardware threads) available to the process.
/// Never returns 0: falls back to 1 when the value cannot be determined.
SOFA_HELPER_API unsigned getLogicalCoreCount();

/// @brief Number of physical CPU cores on the system, i.e. not counting SMT
/// (hyper-threading) siblings.
///
/// Uses the OS topology information (sysctl on macOS, sysfs on Linux, the
/// processor-relationship API on Windows). When the physical count cannot be
/// determined, the logical count is returned instead. Never returns 0.
SOFA_HELPER_API unsigned getPhysicalCoreCount();

/// @brief Number of physical cores belonging to the fastest core type of the CPU.
///
/// On heterogeneous CPUs (Apple silicon, ARM big.LITTLE, Intel P/E cores) this
/// excludes the efficiency cores, which is the right count for work split into
/// equal chunks: a chunk landing on a slow core would otherwise delay the whole
/// parallel section. On homogeneous CPUs, or when the information is
/// unavailable, this equals getPhysicalCoreCount(). Never returns 0.
SOFA_HELPER_API unsigned getPerformanceCoreCount();

}
