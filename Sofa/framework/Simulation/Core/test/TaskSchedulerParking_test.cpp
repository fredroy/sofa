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
#include <sofa/simulation/task/DefaultTaskScheduler.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/task/ParallelForEach.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace sofa
{

// Workers park (block) after an idle period longer than their grace period, and must be
// woken when the next parallel section starts. A lost wake-up shows up as a hang (the owner
// waits for a task nobody runs) or, at best, as a section run by the owner alone. This test
// alternates idle gaps longer than the grace period with parallel sections and checks that
// every section completes and that workers took part in them.
TEST(TaskSchedulerParking, sectionsAfterIdleGapsComplete)
{
    auto* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(simulation::DefaultTaskScheduler::name());
    ASSERT_NE(scheduler, nullptr);
    scheduler->init(0);
    if (scheduler->getThreadCount() < 2)
    {
        GTEST_SKIP() << "scheduler has fewer than 2 threads; nothing to park";
    }

    constexpr int iterations = 40;
    constexpr std::size_t nbElements = 1 << 16;
    std::vector<int> data(nbElements, 0);
    std::atomic<unsigned> rangesRunByWorkers { 0 };
    const auto mainThreadId = std::this_thread::get_id();

    for (int iter = 1; iter <= iterations; ++iter)
    {
        // longer than the workers' idle policy (spin, then about a millisecond of yielding,
        // see WorkerThread.cpp): they park
        std::this_thread::sleep_for(std::chrono::milliseconds(25));

        simulation::parallelForEachRange(*scheduler, std::size_t(0), nbElements,
            [&data, &rangesRunByWorkers, mainThreadId, iter](const simulation::Range<std::size_t>& r)
            {
                if (std::this_thread::get_id() != mainThreadId)
                {
                    rangesRunByWorkers.fetch_add(1, std::memory_order_relaxed);
                }
                for (auto i = r.start; i < r.end; ++i)
                {
                    data[i] = iter;
                }
            });

        for (std::size_t i = 0; i < nbElements; i += 4099)
        {
            ASSERT_EQ(data[i], iter) << "iteration " << iter << ": section did not complete";
        }
    }

    // With several ranges per thread and a section of this size, the woken workers must have
    // taken a share of the work. If they never do, wake-ups are broken (or far too slow).
    EXPECT_GT(rangesRunByWorkers.load(), 0u) << "workers never ran a range after being parked";

    scheduler->stop();
}

// Stopping the scheduler while workers are parked must not hang: stop() has to wake them
// so that they observe the shutdown.
TEST(TaskSchedulerParking, stopWhileParked)
{
    const auto scheduler = std::unique_ptr<simulation::TaskScheduler>(
        simulation::MainTaskSchedulerFactory::instantiate(simulation::DefaultTaskScheduler::name()));
    scheduler->init(4);

    // run one section so that the workers have been woken at least once, then let them park
    simulation::CpuTaskStatus status;
    scheduler->addTask(status, [] {});
    scheduler->workUntilDone(&status);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    const auto t0 = std::chrono::steady_clock::now();
    scheduler->stop();
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_LT(elapsed, std::chrono::seconds(2)) << "stop() took too long: parked workers were not woken";
}

} // namespace sofa
