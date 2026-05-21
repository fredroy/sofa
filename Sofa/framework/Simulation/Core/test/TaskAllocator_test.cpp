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
#include <sofa/simulation/CpuTask.h>
#include <sofa/simulation/CpuTaskStatus.h>
#include <sofa/simulation/Task.h>

namespace sofa
{

// Reproduction for a null-pointer dereference in Task::operator new when
// Task::_allocator has not been initialized.
//
// Task::_allocator is a process-global Task::Allocator* that starts as
// nullptr. The only code path that sets it is
// MainTaskSchedulerFactory::createInRegistry, called when a scheduler is
// looked up. If a Task subclass is heap-allocated *before* that has ever
// happened, Task::operator new dereferences nullptr and segfaults.
//
// On the fixed code, Task::operator new should fall back to a default
// allocation (e.g. ::operator new) when _allocator is null, rather than
// crashing.
//
// This test temporarily clears _allocator (saving and restoring it so other
// tests in the same binary aren't affected) and exercises the heap-Task
// path through addTask(Status&, std::function), which internally does
// `new CallableTask(...)`.

namespace
{

class TrivialTask : public simulation::CpuTask
{
public:
    explicit TrivialTask(simulation::CpuTask::Status* status)
        : simulation::CpuTask(status) {}

    sofa::simulation::Task::MemoryAlloc run() final
    {
        return sofa::simulation::Task::MemoryAlloc::Dynamic;
    }
};

} // namespace

TEST(TaskAllocator, OperatorNewSurvivesNullAllocator)
{
    // Save the current allocator so we can restore it at the end.
    simulation::Task::Allocator* const previous = simulation::Task::getAllocator();
    simulation::Task::setAllocator(nullptr);

    // The fix should make this work without crashing. On the buggy code
    // the next line dereferences a null Task::_allocator inside
    // Task::operator new and aborts the process.
    auto* task = new TrivialTask(nullptr);
    EXPECT_NE(task, nullptr);

    // Free it back through the same allocator-aware path. With a null
    // allocator and the fix in place, operator delete must also fall
    // through to a default deallocation.
    delete task;

    // Restore so subsequent tests share the prior allocator state.
    simulation::Task::setAllocator(previous);
}

} // namespace sofa
