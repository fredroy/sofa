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
#include <sofa/simulation/task/ParallelForEach.h>

#include <atomic>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

namespace sofa
{

// DefaultTaskScheduler takes tasks from any thread, and can be initialized and stopped
// from a thread that did not create it (e.g. a simulation stepping on a thread of its
// own while the GUI thread loads the scenes).

namespace
{

using SchedulerPtr = std::unique_ptr<simulation::TaskScheduler>;

/// A scheduler of its own, outside the registry, so that each test controls on which
/// thread it is created. MainTaskSchedulerFactory::createInRegistry() also installs
/// its task allocator, used to allocate the tasks of parallelForEach.
SchedulerPtr createScheduler()
{
    SchedulerPtr scheduler(simulation::DefaultTaskScheduler::create());
    simulation::Task::setAllocator(scheduler->getTaskAllocator());
    return scheduler;
}

constexpr std::size_t kNbValues = 100000;
constexpr std::size_t kExpectedSum = kNbValues * (kNbValues - 1) / 2;

/// Sum of 0 .. kNbValues-1, computed by parallel tasks.
std::size_t parallelSum(simulation::TaskScheduler& scheduler)
{
    std::vector<std::size_t> values(kNbValues);
    std::iota(values.begin(), values.end(), std::size_t{0});

    std::atomic<std::size_t> sum{0};
    simulation::parallelForEach(scheduler, values.begin(), values.end(),
        [&sum](const std::size_t value) { sum.fetch_add(value, std::memory_order_relaxed); });
    return sum.load();
}

} // namespace

TEST(TaskSchedulerAnyThread, TasksFromAnotherThread)
{
    auto scheduler = createScheduler();
    scheduler->init(4);

    std::size_t sum = 0;
    std::thread other([&scheduler, &sum] { sum = parallelSum(*scheduler); });
    other.join();
    EXPECT_EQ(sum, kExpectedSum);

    // The creating thread still works.
    EXPECT_EQ(parallelSum(*scheduler), kExpectedSum);
    scheduler->stop();
}

TEST(TaskSchedulerAnyThread, InitAndStopFromAnotherThread)
{
    SchedulerPtr scheduler;
    std::thread creator([&scheduler] { scheduler = createScheduler(); });
    creator.join();

    scheduler->init(4);
    EXPECT_EQ(scheduler->getThreadCount(), 4u);
    EXPECT_EQ(parallelSum(*scheduler), kExpectedSum);

    // Another thread count restarts the workers: stop() and start() on this thread.
    scheduler->init(2);
    EXPECT_EQ(scheduler->getThreadCount(), 2u);
    EXPECT_EQ(parallelSum(*scheduler), kExpectedSum);

    scheduler->stop();
}

TEST(TaskSchedulerAnyThread, SuccessiveThreads)
{
    auto scheduler = createScheduler();
    scheduler->init(4);

    // As a simulation thread started and stopped again and again.
    for (int i = 0; i < 20; ++i)
    {
        std::size_t sum = 0;
        std::thread other([&scheduler, &sum] { sum = parallelSum(*scheduler); });
        other.join();
        EXPECT_EQ(sum, kExpectedSum);
    }
    scheduler->stop();
}

TEST(TaskSchedulerAnyThread, ConcurrentThreads)
{
    auto scheduler = createScheduler();
    scheduler->init(4);

    // Threads registering and adding tasks while the workers steal from the others.
    constexpr int kNbThreads = 4;
    constexpr int kNbSumsPerThread = 50;
    std::atomic<int> nbCorrectSums{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kNbThreads; ++t)
    {
        threads.emplace_back([&scheduler, &nbCorrectSums]
        {
            for (int i = 0; i < kNbSumsPerThread; ++i)
            {
                if (parallelSum(*scheduler) == kExpectedSum)
                {
                    nbCorrectSums.fetch_add(1);
                }
            }
        });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
    EXPECT_EQ(nbCorrectSums.load(), kNbThreads * kNbSumsPerThread);
    scheduler->stop();
}

TEST(TaskSchedulerAnyThread, CurrentThreadName)
{
    auto scheduler = createScheduler();
    scheduler->init(2);

    // Formerly a null dereference on a thread the scheduler did not know.
    std::string name;
    std::thread other([&scheduler, &name] { name = scheduler->getCurrentThreadName(); });
    other.join();
    EXPECT_FALSE(name.empty());
    scheduler->stop();
}

} // namespace sofa
