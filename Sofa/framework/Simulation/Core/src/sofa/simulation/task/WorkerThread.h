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

#include <sofa/simulation/config.h>

#include <sofa/simulation/task/Task.h>
#include <sofa/simulation/Locks.h>

#include <thread>
#include <deque>
#include <string>

namespace sofa::simulation
{

class DefaultTaskScheduler;
class Task;

class SOFA_SIMULATION_CORE_API WorkerThread
{
public:

    WorkerThread(DefaultTaskScheduler* const& taskScheduler, int index, const std::string& name = "Worker");

    ~WorkerThread();

    // queue task if there is space, and run it otherwise
    bool addTask(Task* pTask);

    void workUntilDone(Task::Status* status);

    const Task::Status* getCurrentStatus() const { return m_currentStatus; }

    const char* getName() const { return m_name.c_str(); }

    int getType() const { return m_type; }

    const std::thread::id getId() const;

    const std::deque<Task*>* getTasksQueue() { return &m_tasks; }

    std::uint64_t getTaskCount() const { return static_cast<std::uint64_t>(m_taskCount.load(std::memory_order_relaxed)); }

private:

    bool start(DefaultTaskScheduler* const& taskScheduler);

    std::thread* create_and_attach(DefaultTaskScheduler* const& taskScheduler);

    void runTask(Task* task);

    // queue task if there is space (or do nothing)
    bool pushTask(Task* pTask);

    // pop task from queue
    bool popTask(Task** ppTask);

    // steal a task from another thread's queue
    bool stealTask(Task** task);

    /// Per-thread pseudo-random number (xorshift32), used to pick the first victim to steal from
    unsigned nextRandom();

    /// Run queued and stolen tasks until none is available (or status is no longer busy).
    /// @return true if at least one task was executed
    bool doWork(Task::Status* status);

    /// Called after an iteration that found no task: spin, then yield, then sleep
    static void backoff(unsigned idleIterations);

    // thread main loop
    void run(void);

    //void	ThreadProc(void);
    void	Idle(void);

    bool isFinished() const;

    enum
    {
        Max_TasksPerThread = 256
    };

    const std::string m_name;

    const int m_type;

    /// Position of this thread in DefaultTaskScheduler::m_workers (0 is the main thread)
    const unsigned m_index;

    unsigned m_randomState;

    simulation::SpinLock m_taskMutex;

    std::deque<Task*> m_tasks;

    /// Mirror of m_tasks.size(), maintained under m_taskMutex. Read without the lock by
    /// thieves and by the owner to skip empty queues without any lock traffic.
    std::atomic<int> m_taskCount { 0 };

    std::thread  m_stdThread;

    Task::Status*	m_currentStatus;

    DefaultTaskScheduler*     m_taskScheduler;

    // The following members may be accessed by _multiple_ threads at the same time:
    std::atomic<bool>	m_finished;

    friend class DefaultTaskScheduler;
};

} // namespace sofa::simulation
