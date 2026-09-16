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
    // acquire: a thread that observes the counter at zero must see every write the tasks made
    // before their release decrement in setBusy(false)
    return (m_busy.load(std::memory_order_acquire) > 0);
}

int CpuTaskStatus::setBusy(bool busy)
{
    if (busy)
    {
        return m_busy.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        // release: publishes the task's writes to whoever observes the counter reaching zero.
        // acq_rel rather than release so that the last decrement also sees the writes of the
        // other tasks (same pattern as a shared_ptr reference count).
        return m_busy.fetch_sub(1, std::memory_order_acq_rel);
    }
}
}
