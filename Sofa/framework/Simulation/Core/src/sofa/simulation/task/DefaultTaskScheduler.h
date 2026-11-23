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
#include <map>
#include <string> 
#include <mutex>
#include <atomic>
#include <cstdint>
#include <vector>


namespace sofa::simulation
{

class WorkerThread;

/**
 * Task scheduler with a pool of worker threads.
 *
 * Tasks can be added from any thread, not only from the one that created the scheduler:
 * a thread is registered on its first use, and its tasks can be stolen by the workers
 * like the ones of the creating thread. init() and stop() can be called from any thread
 * too.
 */
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

    /// The WorkerThread of the calling thread, registered on its first call.
    WorkerThread* getCurrent();
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
            
    bool isInitialized() const { return m_isInitialized; }
            
    bool isClosing() const { return m_isClosing.load(std::memory_order_acquire); }
            
    void	WaitForWorkersToBeReady();
            
    void	wakeUpWorkers();
            
    WorkerThread* getCurrentThread();
            
    WorkerThread* getWorkerThread(const std::thread::id id);

            
    static const std::string _name;

    /// The workers of the pool, and the other threads that used the scheduler (the one
    /// that created it, and the threads registered by getCurrent()), which own no
    /// std::thread. Guarded by m_threadsMutex.
    std::map< std::thread::id, WorkerThread*> _threads;
    std::mutex m_threadsMutex;

    /// The threads of _threads, which the workers go through to steal tasks, without
    /// lock, while other threads register: a new list is published on each change, and
    /// the previous ones are kept until the scheduler is destroyed.
    std::atomic<const std::vector<WorkerThread*>*> m_threadList{nullptr};
    std::vector<std::unique_ptr<const std::vector<WorkerThread*>>> m_threadLists;
    /// Publishes the content of _threads. With m_threadsMutex.
    void publishThreadList();
    const std::vector<WorkerThread*>& getThreadList() const { return *m_threadList.load(std::memory_order_acquire); }

    /// Unique among the schedulers created by the process: the threads remember their
    /// WorkerThread for this id (getCurrent).
    const std::uint64_t m_id;
    /// Threads registered by getCurrent(), to name them. With m_threadsMutex.
    unsigned m_registeredThreadCount{0};

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
            
    bool m_isInitialized;

    unsigned m_workerThreadCount;
            
    std::atomic<bool> m_workerThreadsIdle;
            
    std::atomic<bool> m_isClosing;
            
    unsigned m_threadCount;
            
    friend class WorkerThread;
};

} // namespace sofa::simulation
