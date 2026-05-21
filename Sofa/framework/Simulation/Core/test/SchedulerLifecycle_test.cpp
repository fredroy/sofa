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
#include <sofa/simulation/CpuTask.h>
#include <sofa/simulation/CpuTaskStatus.h>
#include <sofa/simulation/DefaultTaskScheduler.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>

#include <atomic>

namespace sofa
{

// Regression guard for the scheduler's lifecycle: init / stop / init.
//
// The scheduler stores its state across several flags (m_isInitialized,
// m_isClosing, m_workerThreadsIdle, m_threadCount) that have all been
// written and read across threads in subtle ways. The cleanup that
// accompanies this test makes them properly atomic and removes the
// vestigial WaitForWorkersToBeReady; this test pins the public contract
// that the scheduler can be torn down and rebuilt cleanly while still
// dispatching work between cycles.

namespace
{

void dispatchOneTrivialBurst(simulation::TaskScheduler& scheduler, int n)
{
    std::atomic<int> counter { 0 };
    simulation::CpuTaskStatus status;
    for (int i = 0; i < n; ++i)
    {
        scheduler.addTask(status, [&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    scheduler.workUntilDone(&status);
    ASSERT_EQ(counter.load(std::memory_order_relaxed), n);
}

} // namespace

// init -> use -> stop -> init -> use -> stop -> init -> use should all
// work without crash, deadlock, or stale state. We deliberately use
// different thread counts on each init() call to exercise the
// stop+restart path inside DefaultTaskScheduler::init.
TEST(SchedulerLifecycle, InitStopInitDispatchesCorrectly)
{
    auto* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(
        simulation::DefaultTaskScheduler::name());
    ASSERT_NE(scheduler, nullptr);

    // Cycle 1: default thread count.
    scheduler->init(0);
    if (scheduler->getThreadCount() < 2)
    {
        GTEST_SKIP() << "scheduler has fewer than 2 threads; nothing to verify";
    }
    dispatchOneTrivialBurst(*scheduler, 64);
    scheduler->stop();

    // Cycle 2: explicit thread count.
    scheduler->init(2);
    EXPECT_EQ(scheduler->getThreadCount(), 2u);
    dispatchOneTrivialBurst(*scheduler, 64);
    scheduler->stop();

    // Cycle 3: back to default. Ensures we don't carry stale per-thread
    // state from cycle 2.
    scheduler->init(0);
    dispatchOneTrivialBurst(*scheduler, 64);
    scheduler->stop();
}

// init -> stop -> stop should be a no-op the second time (idempotent).
// On the buggy code, double-stop walked _threads and freed already-freed
// workers; we want to keep that path safe forever.
TEST(SchedulerLifecycle, DoubleStopIsHarmless)
{
    auto* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(
        simulation::DefaultTaskScheduler::name());
    ASSERT_NE(scheduler, nullptr);

    scheduler->init(0);
    if (scheduler->getThreadCount() < 2)
    {
        GTEST_SKIP() << "scheduler has fewer than 2 threads; nothing to verify";
    }

    scheduler->stop();
    // Second stop must not crash, deadlock, or double-free.
    scheduler->stop();

    // Re-init must still work after the duplicated stop().
    scheduler->init(0);
    dispatchOneTrivialBurst(*scheduler, 32);
    scheduler->stop();
}

} // namespace sofa
