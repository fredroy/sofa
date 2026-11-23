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
#include <sofa/simulation/task/DefaultTaskScheduler.h>

#include <sofa/helper/system/thread/thread_specific_ptr.h>
#include <sofa/simulation/task/WorkerThread.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>

#include <chrono>
#include <vector>

namespace sofa::simulation
{

const bool DefaultTaskSchedulerRegistered = MainTaskSchedulerFactory::registerScheduler(
    DefaultTaskScheduler::name(),
    &DefaultTaskScheduler::create);

class StdTaskAllocator : public Task::Allocator
{
public:

    void* allocate(std::size_t sz) final
    {
        return ::operator new(sz);
    }

    void free(void* ptr, std::size_t sz) final
    {
        SOFA_UNUSED(sz);
        ::operator delete(ptr);
    }
};

namespace
{
    std::atomic<std::uint64_t> s_nextSchedulerId{1};

    /// The WorkerThread of the calling thread in the scheduler it used last, to find it
    /// without lock.
    struct CurrentThreadCache
    {
        std::uint64_t schedulerId{0};
        WorkerThread* thread{nullptr};
    };
    thread_local CurrentThreadCache t_currentThread;
}

DefaultTaskScheduler* DefaultTaskScheduler::create()
{
    return new DefaultTaskScheduler();
}

DefaultTaskScheduler::DefaultTaskScheduler()
: TaskScheduler()
, m_id(s_nextSchedulerId.fetch_add(1))
{
    m_isInitialized = false;
    m_threadCount = 0;
    m_isClosing = false;

    // The creating thread
    std::lock_guard lock(m_threadsMutex);
    _threads[std::this_thread::get_id()] = new WorkerThread(this, 0, "Main  ");
    publishThreadList();
}

DefaultTaskScheduler::~DefaultTaskScheduler()
{
    if ( m_isInitialized )
    {
        stop();
    }

    // Only the threads that are not workers are left: the creating thread and the
    // registered ones, which own no std::thread.
    std::lock_guard lock(m_threadsMutex);
    for (const auto& [threadId, thread] : _threads)
    {
        delete thread;
    }
    _threads.clear();
    m_threadList.store(nullptr);
}

void DefaultTaskScheduler::publishThreadList()
{
    auto list = std::make_unique<std::vector<WorkerThread*>>();
    list->reserve(_threads.size());
    for (const auto& [threadId, thread] : _threads)
    {
        list->push_back(thread);
    }
    m_threadList.store(list.get(), std::memory_order_release);
    m_threadLists.push_back(std::move(list));
}

WorkerThread* DefaultTaskScheduler::getWorkerThread(const std::thread::id id)
{
    std::lock_guard lock(m_threadsMutex);
    const auto thread =_threads.find(id);
    if (thread == _threads.end() )
    {
        return nullptr;
    }
    return thread->second;
}

Task::Allocator* DefaultTaskScheduler::getTaskAllocator()
{
    static StdTaskAllocator defaultTaskAllocator;
    return &defaultTaskAllocator;
}

void DefaultTaskScheduler::init(const unsigned int NbThread )
{
    if ( m_isInitialized )
    {
        if ( (NbThread == m_threadCount) || (NbThread==0 && m_threadCount==GetHardwareThreadsCount()) )
        {
            return;
        }
        stop();
    }

    start(NbThread);
}

void DefaultTaskScheduler::start(const unsigned int NbThread )
{
    stop();

    m_isClosing = false;
    m_workerThreadsIdle = true;
    m_mainTaskStatus	= nullptr;

    // default number of thread: only physical cores. no advantage from hyperthreading.
    m_threadCount = GetHardwareThreadsCount();

    if ( NbThread > 0 )//&& NbThread <= MAX_THREADS  )
    {
        m_threadCount = NbThread;
    }

    /* start worker threads */
    for( unsigned int i=1; i<m_threadCount; ++i)
    {
        WorkerThread* thread = new WorkerThread(this, int(i));
        {
            // Registered before it can look itself up (getCurrent).
            std::lock_guard lock(m_threadsMutex);
            thread->create_and_attach(this);
            _threads[thread->getId()] = thread;
            publishThreadList();
        }
        thread->start(this);
    }

    m_workerThreadCount = m_threadCount;
    m_isInitialized = true;
}



void DefaultTaskScheduler::stop()
{
    m_isClosing = true;

    if ( m_isInitialized )
    {
        // wait for all
        WaitForWorkersToBeReady();
        wakeUpWorkers();
        m_isInitialized = false;

        // The workers are the threads owning a std::thread; the others (the creating
        // thread, the registered ones) stay registered, whichever thread calls stop().
        std::vector<WorkerThread*> workers;
        {
            std::lock_guard lock(m_threadsMutex);
            for (const auto& [threadId, thread] : _threads)
            {
                if (thread->m_stdThread.joinable())
                {
                    workers.push_back(thread);
                }
            }
        }

        // All of them finish before any is deleted: a running worker may still be
        // stealing from the others.
        for (const WorkerThread* workerThread : workers)
        {
            // cpu busy wait
            while (!workerThread->isFinished())
            {
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        {
            std::lock_guard lock(m_threadsMutex);
            for (const WorkerThread* workerThread : workers)
            {
                _threads.erase(workerThread->getId());
            }
            publishThreadList();
        }

        // free memory (joins the std::thread)
        for (const WorkerThread* workerThread : workers)
        {
            delete workerThread;
        }

        m_threadCount = 1;
        m_workerThreadCount = 1;
    }
}

WorkerThread* DefaultTaskScheduler::getCurrent()
{
    if (t_currentThread.schedulerId == m_id)
    {
        return t_currentThread.thread;
    }

    const auto threadId = std::this_thread::get_id();
    std::lock_guard lock(m_threadsMutex);
    auto [it, inserted] = _threads.try_emplace(threadId, nullptr);
    if (inserted)
    {
        // A thread that never used the scheduler: registered, so that the workers can
        // steal its tasks. It stays registered until the scheduler is destroyed (a later
        // thread that gets the same id reuses it).
        it->second = new WorkerThread(this, int(++m_registeredThreadCount), "Registered ");
        publishThreadList();
    }
    t_currentThread = { m_id, it->second };
    return it->second;
}

const char* DefaultTaskScheduler::getCurrentThreadName()
{
    const WorkerThread* thread = getCurrent();
    return thread->getName();
}

int DefaultTaskScheduler::getCurrentThreadType()
{
    const WorkerThread* thread = getCurrent();
    return thread->getType();
}

bool DefaultTaskScheduler::addTask(Task* task)
{
    WorkerThread* thread = getCurrent();
    return thread->addTask(task);
}

void DefaultTaskScheduler::workUntilDone(Task::Status* status)
{
    WorkerThread* thread = getCurrent();
    thread->workUntilDone(status);
}

void DefaultTaskScheduler::wakeUpWorkers()
{
    {
        std::lock_guard guard(m_wakeUpMutex);
        m_workerThreadsIdle = false;
    }
    m_wakeUpEvent.notify_all();
}

void DefaultTaskScheduler::WaitForWorkersToBeReady()
{
    m_workerThreadsIdle = true;
}

void DefaultTaskScheduler::setMainTaskStatus(const Task::Status* mainTaskStatus)
{
    m_mainTaskStatus.store(mainTaskStatus, std::memory_order_relaxed);
}

bool DefaultTaskScheduler::testMainTaskStatus(const Task::Status* status)
{
    return m_mainTaskStatus.load(std::memory_order_relaxed) == status;
}

} // namespace sofa::simulation
