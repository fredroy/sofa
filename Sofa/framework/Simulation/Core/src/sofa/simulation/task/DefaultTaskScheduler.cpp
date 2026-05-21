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

// Per-thread pointer to the WorkerThread that "is" this thread. Set by
// each WorkerThread on entry to run(), and by the scheduler constructor
// for the main thread. Looked up by getCurrent() to avoid an
// unsynchronized std::map lookup on the hot path.
//
// Caveat: the value is process-global (one slot per OS thread, shared
// across all DefaultTaskScheduler instances). In practice SOFA only
// instantiates one scheduler at a time via MainTaskSchedulerFactory. If
// two distinct DefaultTaskSchedulers were created on the same thread,
// the most recent constructor would overwrite the previous main-thread
// registration. The previous std::map design had similar surprises and
// any defensive solution would re-introduce per-call lookup cost.
namespace
{
thread_local WorkerThread* t_currentWorker = nullptr;
}

void DefaultTaskScheduler::setCurrentWorkerForThisThread(WorkerThread* worker)
{
    t_currentWorker = worker;
}

DefaultTaskScheduler::DefaultTaskScheduler()
: TaskScheduler()
{
    m_isInitialized.store(false, std::memory_order_relaxed);
    m_threadCount = 0;
    m_isClosing.store(false, std::memory_order_relaxed);

    // The thread that constructs the scheduler becomes its "main" worker
    // thread. The WorkerThread is owned by this scheduler and survives
    // stop()/start() cycles; only worker pool members are recreated.
    m_mainThread = new WorkerThread(this, 0, "Main  ");
    setCurrentWorkerForThisThread(m_mainThread);
}

DefaultTaskScheduler::~DefaultTaskScheduler()
{
    if (isInitialized())
    {
        stop();
    }
    // Drop the thread_local registration only if it still points at our
    // main worker (a different scheduler may have taken over since).
    if (t_currentWorker == m_mainThread)
    {
        setCurrentWorkerForThisThread(nullptr);
    }
    delete m_mainThread;
    m_mainThread = nullptr;
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
    }

    return;
}

WorkerThread* DefaultTaskScheduler::getCurrent()
{
    // Hot-path lookup: the calling thread either is a worker that
    // registered itself in WorkerThread::run, or it's the main thread
    // that the scheduler constructor registered. If neither, t_currentWorker
    // is nullptr (e.g. a user-spawned std::thread calling addTask) and the
    // caller will get a clear nullptr that signals "this thread is not
    // associated with the scheduler".
    return t_currentWorker;
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
