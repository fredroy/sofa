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
#include <sofa/simulation/task/TaskScheduler.h>

#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/task/MainTaskSchedulerRegistry.h>

#include <sofa/helper/system/Cpu.h>

namespace sofa::simulation
{
unsigned TaskScheduler::GetHardwareThreadsCount()
{
    // Performance cores only:
    // - SMT siblings are excluded because the parallel loops in SOFA are
    //   memory-bound and gain nothing from hyper-threading.
    // - Efficiency cores are excluded because parallelForEach splits the work
    //   into one equal chunk per thread: a chunk running on a slow core delays
    //   the whole section. Measured on an Apple M3 Max (10P + 4E), 14 threads
    //   were 1.6x to 3.5x slower than 10 on FEM scenes.
    // Never returns 0.
    return sofa::helper::system::getPerformanceCoreCount();
}

bool TaskScheduler::addTask(Task::Status& status, const std::function<void()>& task)
{
    class CallableTask final : public Task
    {
    public:
        CallableTask(int scheduledThread, Task::Status& status, std::function<void()> task)
            : Task(scheduledThread)
            , m_status(status)
            , m_task(std::move(task))
        {}
        ~CallableTask() override = default;
        sofa::simulation::Task::MemoryAlloc run() final
        {
            m_task();
            return MemoryAlloc::Dynamic;
        }

        Task::Status* getStatus() const override
        {
            return &m_status;
        }

    private:
        Task::Status& m_status;
        std::function<void()> m_task;
    };

    return addTask(new CallableTask(-1, status, task)); //destructor should be called after run() because it returns MemoryAlloc::Dynamic
}

} // namespace sofa::simulation
