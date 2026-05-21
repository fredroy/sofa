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

#include <sofa/config.h>

#include <sofa/simulation/task/TaskScheduler.h>

// default
#include <thread>
#include <condition_variable>
#include <memory>
#include <string>
#include <mutex>
#include <atomic>
#include <vector>


namespace sofa::simulation
{

class WorkerThread;

class SOFA_SIMULATION_CORE_API DefaultTaskScheduler : public TaskScheduler
{
    enum
    {
        MAX_THREADS = 16,
        STACKSIZE = 64 * 1024 /* 64K */,
    };
            
public:
            
    // interface

    /**
     * Call stop() and start() if not already initialized
     * @param nbThread
     */
    virtual void init(const unsigned int nbThread = 0) final;

    /**
     * Wait and destroy worker threads
     */
    void stop() final;

    WorkerThread* getCurrent();

    /// Called by each WorkerThread on entry to run() (and by the scheduler
    /// constructor for the main thread) to register itself so that
    /// getCurrent() can return it via a thread_local without searching a
    /// map. Pass nullptr on thread exit.
    static void setCurrentWorkerForThisThread(WorkerThread* worker);

    unsigned int getThreadCount(void)  const final { return m_threadCount; }
    const char* getCurrentThreadName() override final;
    int getCurrentThreadType() override final;

    // queue task if there is space, and run it otherwise
    bool addTask(Task* task) override final;
    void workUntilDone(Task::Status* status) override final;
    Task::Allocator* getTaskAllocator() override final;

    // factory methods: name, creator function
    static const char* name() { return "_default"; }
            
    static DefaultTaskScheduler* create();

private:
            
    bool isInitialized() const { return m_isInitialized.load(std::memory_order_relaxed); }

    bool isClosing() const { return m_isClosing.load(std::memory_order_relaxed); }
            
    void	wakeUpWorkers();
            
    WorkerThread* getCurrentThread();

    static const std::string _name;

    /// Snapshot of the worker threads (excluding the main thread) populated
    /// at start() and cleared at stop(). Iterated by WorkerThread::stealTask
    /// without synchronization: it is only mutated when no workers are
    /// running, so concurrent reads from the worker pool see a stable
    /// vector.
    std::vector<WorkerThread*> m_workers;

    /// The WorkerThread of the thread that constructed the scheduler.
    /// The main thread is the typical producer in SOFA's
    /// parallelForEachRange pattern (addTask runs on it and tasks land in
    /// its m_tasks queue), so worker threads must be able to steal from
    /// it; otherwise the parallel sweep degenerates to running every
    /// task on the producing thread.
    /// Owned by this scheduler. Survives stop()/start() cycles.
    WorkerThread* m_mainThread { nullptr };

    std::atomic<const Task::Status*> m_mainTaskStatus;
    void setMainTaskStatus(const Task::Status* mainTaskStatus);
    bool testMainTaskStatus(const Task::Status*);
            
    std::mutex  m_wakeUpMutex;
            
    std::condition_variable m_wakeUpEvent;
            
    DefaultTaskScheduler();
            
    DefaultTaskScheduler(const DefaultTaskScheduler&) = delete;
            
    ~DefaultTaskScheduler() override;

    /**
     * Create worker threads
     * If the number of required threads is 0, the number of threads will be equal to the
     * result of GetHardwareThreadsCount()
     *
     * @param NbThread
     */
    void start(unsigned int NbThread);
            
    // Lifecycle flags read by worker threads (run() polls isClosing()) and
    // written by start()/stop() on the main thread. Atomic to satisfy the
    // C++ memory model contract; relaxed ordering is sufficient because
    // they coordinate liveness, not data publication (the actual data
    // synchronization goes through m_wakeUpMutex / m_wakeUpEvent and
    // m_taskMutex inside WorkerThread).
    std::atomic<bool> m_isInitialized;

    unsigned m_workerThreadCount;

    std::atomic<bool> m_workerThreadsIdle;

    std::atomic<bool> m_isClosing;
            
    unsigned m_threadCount;
            
    friend class WorkerThread;
};

} // namespace sofa::simulation
