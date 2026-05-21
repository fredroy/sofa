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

DefaultTaskScheduler* DefaultTaskScheduler::create()
{
    return new DefaultTaskScheduler();
}

DefaultTaskScheduler::DefaultTaskScheduler()
: TaskScheduler()
{
    m_isInitialized.store(false, std::memory_order_relaxed);
    m_threadCount = 0;
    m_isClosing.store(false, std::memory_order_relaxed);

    // init global static thread local var
    {
        _threads[std::this_thread::get_id()] = new WorkerThread(this, 0, "Main  ");// new WorkerThread(this, 0, "Main  ");
    }
}

DefaultTaskScheduler::~DefaultTaskScheduler()
{
    if (isInitialized())
    {
        stop();
    }
}

WorkerThread* DefaultTaskScheduler::getWorkerThread(const std::thread::id id)
{
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
    if (isInitialized())
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

    m_isClosing.store(false, std::memory_order_relaxed);
    m_workerThreadsIdle = true;
    m_mainTaskStatus	= nullptr;

    // default number of thread: only physical cores. no advantage from hyperthreading.
    m_threadCount = GetHardwareThreadsCount();

    if ( NbThread > 0 )//&& NbThread <= MAX_THREADS  )
    {
        m_threadCount = NbThread;
    }

    /* start worker threads */
    m_workers.reserve(m_threadCount > 0 ? m_threadCount - 1 : 0);
    for( unsigned int i=1; i<m_threadCount; ++i)
    {
        WorkerThread* thread = new WorkerThread(this, int(i));
        thread->create_and_attach(this);
        _threads[thread->getId()] = thread;
        m_workers.push_back(thread);
        thread->start(this);
    }

    m_workerThreadCount = m_threadCount;
    m_isInitialized.store(true, std::memory_order_relaxed);
}



void DefaultTaskScheduler::stop()
{
    m_isClosing.store(true, std::memory_order_relaxed);

    if (isInitialized())
    {
        // Wake every worker so they observe m_isClosing == true and exit
        // their run() loop. The wake itself sets m_workerThreadsIdle to
        // false under m_wakeUpMutex.
        wakeUpWorkers();
        m_isInitialized.store(false, std::memory_order_relaxed);

        // ~WorkerThread joins the underlying std::thread, so deleting
        // each worker here blocks until that worker has actually exited
        // its run() loop. Iterate m_workers (which contains only the
        // worker threads, never the main thread) so we don't have to
        // skip-the-main-thread inside the loop.
        for (WorkerThread* worker : m_workers)
        {
            delete worker;
        }
        m_workers.clear();

        m_threadCount = 1;
        m_workerThreadCount = 1;

        const auto mainThreadIt = _threads.find(std::this_thread::get_id());
        WorkerThread* mainThread = mainThreadIt->second;
        _threads.clear();
        _threads[std::this_thread::get_id()] = mainThread;
    }

    return;
}

WorkerThread* DefaultTaskScheduler::getCurrent()
{
    return getWorkerThread(std::this_thread::get_id());
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

void DefaultTaskScheduler::setMainTaskStatus(const Task::Status* mainTaskStatus)
{
    m_mainTaskStatus.store(mainTaskStatus, std::memory_order_relaxed);
}

bool DefaultTaskScheduler::testMainTaskStatus(const Task::Status* status)
{
    return m_mainTaskStatus.load(std::memory_order_relaxed) == status;
}

} // namespace sofa::simulation
