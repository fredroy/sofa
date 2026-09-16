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
#include <sofa/simulation/task/CpuTaskStatus.h>
#include <sofa/simulation/task/CpuTask.h>
#include <sofa/simulation/task/DefaultTaskScheduler.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>

namespace sofa
{

// CpuTaskStatus is the completion signal of the task scheduler: a task's results are written
// by a worker, then the worker calls setBusy(false); the waiting thread spins on isBusy() and
// reads the results as soon as it returns false. For this to be correct, every write made
// before setBusy(false) must be visible to a thread that observed isBusy() == false, i.e. the
// decrement must be a release and the load an acquire (the shared_ptr reference-count pattern).
//
// Both tests below are message-passing litmus tests of that contract. A stale payload word is
// a direct observation of the completion signal overtaking the writes it publishes; this is
// hardware and timing dependent (impossible on x86, whose stores stay ordered; allowed on ARM
// but not observed in 9 million iterations on an Apple M3 Max). ThreadSanitizer, however,
// detects the missing happens-before edge deterministically: with relaxed ordering the first
// test reports a data race on the payload read (this file, in firstStaleWord) against the
// producer's write, and with acquire/release it reports nothing.
//
// To run under ThreadSanitizer, the whole build must be instrumented (-fsanitize=thread on
// Sofa.Simulation.Core as well): the second test goes through the scheduler, whose spinlocks
// are what synchronizes the pusher with the worker, and an uninstrumented scheduler library
// makes the sanitizer report races that the spinlocks actually prevent.

namespace
{
constexpr int payloadSize = 64;              // 8 cache lines
using Payload = std::array<std::uint64_t, payloadSize>;

/// @return the index of the first word that does not hold `expected`, or -1
int firstStaleWord(const Payload& payload, const std::uint64_t expected)
{
    for (int k = payloadSize - 1; k >= 0; --k) // the last written words are the most likely to be stale
    {
        if (payload[k] != expected)
        {
            return k;
        }
    }
    return -1;
}
}

TEST(CpuTaskStatus, writesBeforeCompletionAreVisibleToWaiter)
{
    constexpr std::uint64_t iterations = 200000;

    simulation::CpuTaskStatus status;
    alignas(64) Payload payload {};
    alignas(64) std::atomic<std::uint64_t> go { 0 };

    std::thread producer([&]
    {
        for (std::uint64_t iter = 1; iter <= iterations; ++iter)
        {
            while (go.load(std::memory_order_acquire) != iter) {}
            for (auto& word : payload)
            {
                word = iter;              // the "task result"
            }
            status.setBusy(false);        // the completion signal under test
        }
    });

    std::uint64_t staleReads = 0;
    int lastStaleIndex = -1;
    for (std::uint64_t iter = 1; iter <= iterations; ++iter)
    {
        status.setBusy(true);
        go.store(iter, std::memory_order_release);

        while (status.isBusy()) {}        // the wait under test

        const int stale = firstStaleWord(payload, iter);
        if (stale >= 0)
        {
            ++staleReads;
            lastStaleIndex = stale;
        }
    }
    producer.join();

    EXPECT_EQ(staleReads, 0u) << "completion became visible before the writes it publishes"
        " (last stale word index " << lastStaleIndex << ")";
}

// Same contract through the real scheduler: a task run by a worker fills the payload, the main
// thread reads it right after workUntilDone().
TEST(CpuTaskStatus, taskResultsAreVisibleAfterWorkUntilDone)
{
    constexpr std::uint64_t iterations = 20000;

    auto* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(simulation::DefaultTaskScheduler::name());
    ASSERT_NE(scheduler, nullptr);
    scheduler->init(0);
    if (scheduler->getThreadCount() < 2)
    {
        GTEST_SKIP() << "scheduler has fewer than 2 threads; nothing to publish across threads";
    }

    alignas(64) Payload payload {};
    // Set by the task when it starts. The main thread only runs tasks from inside
    // workUntilDone(), so waiting for this flag before calling workUntilDone() guarantees that a
    // worker thread, not the main thread, runs the task. Relaxed on purpose: it must not create
    // the synchronization that the completion signal is supposed to provide.
    alignas(64) std::atomic<std::uint64_t> started { 0 };

    std::uint64_t staleReads = 0;
    int lastStaleIndex = -1;
    for (std::uint64_t iter = 1; iter <= iterations; ++iter)
    {
        simulation::CpuTaskStatus status;
        scheduler->addTask(status, [&payload, &started, iter]
        {
            started.store(iter, std::memory_order_relaxed);
            for (auto& word : payload)
            {
                word = iter;
            }
        });

        while (started.load(std::memory_order_relaxed) != iter) {}   // a worker has taken the task
        scheduler->workUntilDone(&status);

        const int stale = firstStaleWord(payload, iter);
        if (stale >= 0)
        {
            ++staleReads;
            lastStaleIndex = stale;
        }
    }

    EXPECT_EQ(staleReads, 0u) << "task results were not visible after workUntilDone()"
        " (last stale word index " << lastStaleIndex << ")";

    scheduler->stop();
}

} // namespace sofa
