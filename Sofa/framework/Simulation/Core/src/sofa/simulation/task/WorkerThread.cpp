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
#include <sofa/simulation/task/WorkerThread.h>
#include <sofa/simulation/task/DefaultTaskScheduler.h>

#include <cassert>
#include <mutex>
#include <chrono>
#include <thread>

#ifdef WIN32
#include <processthreadsapi.h>
#endif

namespace sofa::simulation
{

namespace
{
// Idle policy of a worker that finds no task, by number of consecutive empty iterations:
//   [0, spinIterations)               pure spin: the next task is probably microseconds away
//   [spinIterations, yieldIterations) yield: give the core to threads that have work
//   [yieldIterations, parkAfter...)   doze: short sleeps (disabled by default, see below)
//   then                              park: block until a section starts (WorkerThread::park)
//
// Dozing (dozeIterations > 0) makes workers poll every dozeSleep instead of parking, so the
// first section after an idle gap does not pay for waking cores from deep idle: on an Apple
// M3 Max a 135 us compute section run 3 ms after the previous one took 350 us (p90 1.1 ms)
// when parking right after the yield phase, 200 us (p90 230 us) with 10 ms of dozing. But it
// is a net loss on real scenes: an interactive scene with about 10 ms of serial constraint
// solving per step ran 4 to 7% slower with dozing, every phase included, because nine
// workers waking every 50 us disturb the thread doing the serial work. Parking right after
// the yield phase is therefore the default; the knob is kept for latency-critical setups.
constexpr unsigned spinIterations = 64;
constexpr unsigned yieldIterations = 1024;
constexpr auto dozeSleep = std::chrono::microseconds(50);
constexpr unsigned dozeIterations = 0;
constexpr unsigned parkAfterIdleIterations = yieldIterations + dozeIterations;

/// Number of parked workers each waker (section owner or freshly woken worker) wakes in turn.
/// 2 gives a binary wake-up tree: the pool is fully awake after log2(N) wake-up latencies.
constexpr unsigned wakeFanOut = 2;
}

WorkerThread::WorkerThread(DefaultTaskScheduler *const &taskScheduler, const int index, const std::string &name)
        : m_name(name + std::to_string(index))
        , m_type(0)
        , m_index(static_cast<unsigned>(index))
        , m_randomState(0x9E3779B9u * (static_cast<unsigned>(index) + 1u) | 1u) // non-zero seed, different per thread
        , m_tasks()
        , m_taskScheduler(taskScheduler)
{
    assert(taskScheduler);
    m_finished.store(false, std::memory_order_relaxed);
    m_currentStatus = nullptr;
}


WorkerThread::~WorkerThread()
{
    if (m_stdThread.joinable())
    {
        m_stdThread.join();
    }
    m_finished.store(true, std::memory_order_relaxed);
}

bool WorkerThread::isFinished() const
{
    return m_finished.load(std::memory_order_relaxed);
}

bool WorkerThread::start(DefaultTaskScheduler *const &taskScheduler)
{
    SOFA_UNUSED(taskScheduler);
    assert(taskScheduler);
    m_currentStatus = nullptr;

    return true;
}

std::thread *WorkerThread::create_and_attach(DefaultTaskScheduler *const &taskScheduler)
{
    m_taskScheduler = taskScheduler;
    m_stdThread = std::thread([this] { run(); });
    return &m_stdThread;
}

void WorkerThread::run(void)
{
#ifdef WIN32
    const std::wstring widestr = std::wstring(m_name.begin(), m_name.end());
    (void)SetThreadDescription(
        GetCurrentThread(),
        widestr.c_str()
        );
#endif

    //workerThreadIndex = this;
    //TaskSchedulerDefault::_threads[std::this_thread::get_id()] = this;

    // main loop
    while (!m_taskScheduler->isClosing())
    {
        unsigned idleIterations = 0;
        while (!m_taskScheduler->isClosing())
        {
            if (m_taskScheduler->testMainTaskStatus(nullptr))
            {
                // No parallel section in progress. Stay awake (spin, yield, then doze) so
                // that the next section does not pay a wake-up; park only after a real idle
                // period. See the idle policy at the top of this file.
                if (idleIterations >= parkAfterIdleIterations)
                {
                    break;
                }
                backoff(idleIterations++);
            }
            else if (doWork(nullptr))
            {
                idleIterations = 0;
            }
            else
            {
                backoff(idleIterations++);
            }
        }

        if (!m_taskScheduler->isClosing())
        {
            park();

            // Just woken for a section: propagate the wake-up to a couple of parked workers
            // so that the whole pool wakes in logarithmic depth without involving the owner
            if (!m_taskScheduler->testMainTaskStatus(nullptr)
                && m_taskScheduler->m_parkedCount.load(std::memory_order_seq_cst) > 0)
            {
                m_taskScheduler->wakeParkedWorkers(wakeFanOut);
            }
        }
    }

    m_finished.store(true, std::memory_order_relaxed);
}

const std::thread::id WorkerThread::getId() const
{
    return m_stdThread.get_id();
}

void WorkerThread::park()
{
    const unsigned epoch = m_parkEpoch.load(std::memory_order_acquire);

    // Publish "parked" first, then re-check the condition. A pusher does the reverse: it
    // publishes the section (main task status) first, then checks the parked count. With
    // sequentially consistent operations on both sides, at least one of the two sees the
    // other's write, so a section that starts right now either is seen here or wakes us.
    m_parked.store(true, std::memory_order_seq_cst);
    m_taskScheduler->m_parkedCount.fetch_add(1, std::memory_order_seq_cst);

    if (!m_taskScheduler->testMainTaskStatus(nullptr) || m_taskScheduler->isClosing())
    {
        // Un-park ourselves, unless a waker already claimed us (it then also fixed the count)
        if (m_parked.exchange(false, std::memory_order_seq_cst))
        {
            m_taskScheduler->m_parkedCount.fetch_sub(1, std::memory_order_seq_cst);
        }
        return;
    }

    // Blocks until a waker increments the epoch (futex / ulock underneath). If that already
    // happened between the checks above and this call, returns immediately.
    m_parkEpoch.wait(epoch, std::memory_order_acquire);
}

bool WorkerThread::doWork(Task::Status *status)
{
    bool didWork = false;
    for (;;)
    {
        Task *task;

        while (popTask(&task))
        {
            // run task in the queue
            runTask(task);
            didWork = true;

            if (status && !status->isBusy())
                return didWork;
        }

        // check if main work is finished
        if (m_taskScheduler->testMainTaskStatus(nullptr))
            return didWork;

        if (!stealTask(&task))
            return didWork;

        // run the stolen task
        runTask(task);
        didWork = true;
    }
}

void WorkerThread::backoff(const unsigned idleIterations)
{
    // Pure spinning keeps a core busy for nothing and, when every core hosts a
    // spinning thread, steals CPU time from the threads that actually have work.
    // See the idle policy described at the top of this file.
    if (idleIterations < spinIterations)
    {
        return;
    }
    if (idleIterations < yieldIterations)
    {
        std::this_thread::yield();
        return;
    }
    std::this_thread::sleep_for(dozeSleep);
}

void WorkerThread::runTask(Task *task)
{
    Task::Status *prevStatus = m_currentStatus;
    m_currentStatus = task->getStatus();

    {
        if (task->run() & Task::MemoryAlloc::Dynamic)
        {
            // Run the (virtual) destructor before freeing the storage.
            // Skipping this leaks any non-trivially-destructible task
            // members: std::function in CallableTask (the lambda overload
            // of addTask) leaks its internal __func<>; std::shared_ptr,
            // std::vector, etc., never release their owned resources.
            const std::size_t taskSize = sizeof(*task);
            task->~Task();
            task->operator delete(task, taskSize);
        }
    }

    m_currentStatus->setBusy(false);
    m_currentStatus = prevStatus;
}

void WorkerThread::workUntilDone(Task::Status *status)
{
    unsigned idleIterations = 0;
    while (status->isBusy())
    {
        if (doWork(status))
        {
            idleIterations = 0;
        }
        else
        {
            backoff(idleIterations++);
        }
    }

    if (m_taskScheduler->testMainTaskStatus(status))
    {
        // End of the section: workers finish their grace period and park by themselves
        m_taskScheduler->setMainTaskStatus(nullptr);
    }
}


bool WorkerThread::popTask(Task **task)
{
    *task = nullptr;

    // Fast path: nothing to pop, no need to take the lock
    if (m_taskCount.load(std::memory_order_relaxed) == 0)
    {
        return false;
    }

    simulation::ScopedLock lock(m_taskMutex);
    if (!m_tasks.empty())
    {
        *task = m_tasks.back();
        m_tasks.pop_back();
        m_taskCount.store(static_cast<int>(m_tasks.size()), std::memory_order_relaxed);
        return true;
    }
    return false;
}


bool WorkerThread::pushTask(Task *task)
{
    // if we're single threaded return false
    if (m_taskScheduler->getThreadCount() < 2)
    {
        return false;
    }

    // Capture the task's Status* before the task becomes visible to workers.
    // Once m_tasks.push_back(task) runs and the lock is released, a worker
    // can pop the task, run it, and (if run() returns MemoryAlloc::Dynamic)
    // free it. Reading task->getStatus() after that point would dereference
    // freed memory. Status objects are owned by the caller of the dispatch
    // (e.g. a CpuTaskStatus on the originating frame) and are guaranteed to
    // outlive the workUntilDone() that follows the push, so the captured
    // pointer remains valid for the post-publish code below.
    Task::Status* statusForMain = nullptr;
    {
        simulation::ScopedLock lock(m_taskMutex);
        statusForMain = task->getStatus();
        const int taskId = statusForMain->setBusy(true);
        task->m_id = taskId;
        m_tasks.push_back(task);
        m_taskCount.store(static_cast<int>(m_tasks.size()), std::memory_order_relaxed);
    }


    if (m_taskScheduler->testMainTaskStatus(nullptr))
    {
        // First task of a parallel section: publish the section before looking for parked
        // workers (see WorkerThread::park for the ordering argument), then wake a couple of
        // them. The woken workers wake the others (see run()), so the section owner pays at
        // most two wake-up system calls and the fan-out runs on the workers. Workers never
        // park while a section is active, so later pushes have nobody to wake.
        m_taskScheduler->setMainTaskStatus(statusForMain);
        if (m_taskScheduler->m_parkedCount.load(std::memory_order_seq_cst) > 0)
        {
            m_taskScheduler->wakeParkedWorkers(wakeFanOut);
        }
    }

    return true;
}

bool WorkerThread::addTask(Task *task)
{
    if (pushTask(task))
    {
        return true;
    }

    // we are single thread: run the task
    runTask(task);

    return false;
}

unsigned WorkerThread::nextRandom()
{
    unsigned x = m_randomState;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    m_randomState = x;
    return x;
}

bool WorkerThread::stealTask(Task **task)
{
    *task = nullptr;

    const auto& workers = m_taskScheduler->m_workers;
    const std::size_t nbWorkers = workers.size();
    if (nbWorkers < 2)
    {
        return false;
    }

    // Start from a random victim so that concurrent thieves spread over the queues instead
    // of all hammering the first one, then walk the ring.
    const std::size_t first = nextRandom() % nbWorkers;
    for (std::size_t k = 0; k < nbWorkers; ++k)
    {
        WorkerThread* victim = workers[(first + k) % nbWorkers];
        if (victim == this)
        {
            continue;
        }

        // Skip visibly empty queues without touching their lock
        if (victim->m_taskCount.load(std::memory_order_relaxed) == 0)
        {
            continue;
        }

        // The owner (or another thief) is using this queue right now: try the next one
        // rather than spinning on its lock.
        if (!victim->m_taskMutex.try_lock())
        {
            continue;
        }

        if (!victim->m_tasks.empty())
        {
            *task = victim->m_tasks.front();
            victim->m_tasks.pop_front();
            victim->m_taskCount.store(static_cast<int>(victim->m_tasks.size()), std::memory_order_relaxed);
            victim->m_taskMutex.unlock();
            return true;
        }
        victim->m_taskMutex.unlock();
    }

    return false;
}

} // namespace sofa::simulation
