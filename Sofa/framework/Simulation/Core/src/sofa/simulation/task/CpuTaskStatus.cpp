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
#include <sofa/simulation/task/CpuTaskStatus.h>

namespace sofa::simulation
{
CpuTaskStatus::CpuTaskStatus(): m_busy(0)
{}

bool CpuTaskStatus::isBusy() const
{
    // Acquire pairs with the release in setBusy(false): once isBusy()
    // observes 0 (or any value matching the producer's decrement), all
    // memory effects of the just-finished task become visible to the
    // caller of workUntilDone.
    return (m_busy.load(std::memory_order_acquire) > 0);
}

int CpuTaskStatus::setBusy(bool busy)
{
    if (busy)
    {
        // Increment doesn't need to publish anything; relaxed is enough.
        // The decrement establishes the synchronization edge.
        return m_busy.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        // Release: a worker calls this after a task's run() returns. Any
        // memory written by the task body must be visible to the thread
        // that next observes m_busy returning to zero through isBusy().
        return m_busy.fetch_sub(1, std::memory_order_release);
    }
}
}
