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
    m_isInitialized = false;
    m_threadCount = 0;
    m_isClosing = false;

    // init global static thread local var
    {
        WorkerThread* mainThread = new WorkerThread(this, 0, "Main  ");
        _threads[std::this_thread::get_id()] = mainThread;
        m_workers.push_back(mainThread);
    }
}

DefaultTaskScheduler::~DefaultTaskScheduler()
{
    if ( m_isInitialized )
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
    m_parkedCount = 0;
    m_mainTaskStatus	= nullptr;

    // default number of thread: only physical cores. no advantage from hyperthreading.
    m_threadCount = GetHardwareThreadsCount();

    if ( NbThread > 0 )//&& NbThread <= MAX_THREADS  )
    {
        m_threadCount = NbThread;
    }

    /* create the worker objects first so that m_workers is complete before any thread runs */
    m_workers.resize(1); // keep the main thread
    for( unsigned int i=1; i<m_threadCount; ++i)
    {
        m_workers.push_back(new WorkerThread(this, int(i)));
    }

    /* start worker threads */
    for( unsigned int i=1; i<m_threadCount; ++i)
    {
        WorkerThread* thread = m_workers[i];
        thread->create_and_attach(this);
        _threads[thread->getId()] = thread;
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
        // release parked workers so that they observe m_isClosing and exit
        wakeAllParkedWorkers();
        m_isInitialized = false;

        for (auto [threadId, workerThread] : _threads)
        {
            // if this is the main thread continue
            if (std::this_thread::get_id() == threadId)
            {
                continue;
            }

            // cpu busy wait
            while (!workerThread->isFinished())
            {
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // free memory
            // cpu busy wait: thread.joint call
            delete workerThread;
            workerThread = nullptr;
        }

        m_threadCount = 1;
        m_workerThreadCount = 1;

        const auto mainThreadIt = _threads.find(std::this_thread::get_id());
        WorkerThread* mainThread = mainThreadIt->second;
        _threads.clear();
        _threads[std::this_thread::get_id()] = mainThread;
        m_workers.assign(1, mainThread);
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

bool DefaultTaskScheduler::wakeOneParkedWorker()
{
    const std::size_t nbWorkers = m_workers.size();
    const unsigned first = m_wakeCursor.fetch_add(1, std::memory_order_relaxed);
    for (std::size_t k = 0; k < nbWorkers; ++k)
    {
        WorkerThread* worker = m_workers[(first + k) % nbWorkers];
        if (!worker->m_parked.load(std::memory_order_relaxed))
        {
            continue;
        }
        // Claim the worker: only one waker can flip the flag
        if (worker->m_parked.exchange(false, std::memory_order_seq_cst))
        {
            m_parkedCount.fetch_sub(1, std::memory_order_seq_cst);
            worker->m_parkEpoch.fetch_add(1, std::memory_order_release);
            worker->m_parkEpoch.notify_one();
            return true;
        }
    }
    return false;
}

unsigned DefaultTaskScheduler::wakeParkedWorkers(const unsigned count)
{
    unsigned woken = 0;
    while (woken < count && wakeOneParkedWorker())
    {
        ++woken;
    }
    return woken;
}

void DefaultTaskScheduler::wakeAllParkedWorkers()
{
    while (wakeOneParkedWorker()) {}
}

void DefaultTaskScheduler::setMainTaskStatus(const Task::Status* mainTaskStatus)
{
    // seq_cst: paired with the parked-count check in WorkerThread::pushTask and the
    // status re-check in WorkerThread::park (Dekker-style handshake, no lost wake-up)
    m_mainTaskStatus.store(mainTaskStatus, std::memory_order_seq_cst);
}

bool DefaultTaskScheduler::testMainTaskStatus(const Task::Status* status)
{
    return m_mainTaskStatus.load(std::memory_order_seq_cst) == status;
}

} // namespace sofa::simulation
