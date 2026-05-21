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
#include <vector>

namespace sofa
{

// Regression guard for CpuTaskStatus::m_busy memory ordering.
//
// The contract of workUntilDone(&status) is "after I return, every memory
// effect produced by the dispatched tasks is visible to the caller". For
// that to be sound, the worker's setBusy(false) decrement must have
// release semantics and the main thread's isBusy() polling load must
// have acquire semantics; relaxed orderings would break the C++ memory
// model contract.
//
// This test is a behavioral regression guard. On x86/arm64, hardware
// over-orders relaxed atomics, so it would have passed even with the
// previous relaxed-only ordering. Its real value is:
//   1. It locks in the public contract for future maintainers.
//   2. Built with -fsanitize=thread, TSan flags the original relaxed
//      ordering as a race; the fix silences it.

namespace
{

// Plain non-atomic payload written by the task body. If the synchronization
// edge between setBusy(false) (worker) and isBusy()==0 (main) is missing,
// TSan flags this read as a race against the task's write.
struct Payload
{
    int values[4] { 0, 0, 0, 0 };
};

void runManyTasksAndCheckPayload(simulation::TaskScheduler& scheduler,
                                 int kTasksPerCycle,
                                 int kCycles)
{
    for (int c = 0; c < kCycles; ++c)
    {
        std::vector<Payload> payloads(kTasksPerCycle);

        simulation::CpuTaskStatus status;
        for (int i = 0; i < kTasksPerCycle; ++i)
        {
            Payload* p = &payloads[i];
            const int seed = c * 1024 + i;
            scheduler.addTask(status, [p, seed]() {
                // Plain (non-atomic) writes: they only become visible to
                // the main thread through the release/acquire edge built
                // into m_busy.
                p->values[0] = seed;
                p->values[1] = seed + 1;
                p->values[2] = seed + 2;
                p->values[3] = seed + 3;
            });
        }
        scheduler.workUntilDone(&status);

        // After workUntilDone returns, the main thread reads the payloads
        // with no further synchronization. With correct ordering this is
        // safe and observes the values written by the tasks.
        for (int i = 0; i < kTasksPerCycle; ++i)
        {
            const int seed = c * 1024 + i;
            ASSERT_EQ(payloads[i].values[0], seed);
            ASSERT_EQ(payloads[i].values[1], seed + 1);
            ASSERT_EQ(payloads[i].values[2], seed + 2);
            ASSERT_EQ(payloads[i].values[3], seed + 3);
        }
    }
}

} // namespace

TEST(CpuTaskStatusVisibility, PayloadsAreVisibleAfterWorkUntilDone)
{
    auto* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(
        simulation::DefaultTaskScheduler::name());
    ASSERT_NE(scheduler, nullptr);

    scheduler->init(0);
    if (scheduler->getThreadCount() < 2)
    {
        GTEST_SKIP() << "scheduler has fewer than 2 threads; visibility cannot race";
    }

    runManyTasksAndCheckPayload(*scheduler, /*kTasksPerCycle=*/64, /*kCycles=*/200);

    scheduler->stop();
}

} // namespace sofa
