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
#pragma once

#include <sofa/helper/logging/Messaging.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/simulation/task/CpuTaskStatus.h>
#include <sofa/type/vector_T.h>

#include <algorithm>
#include <vector>

namespace sofa::simulation
{

/**
 * Default number of ranges generated per scheduler thread by parallelForEachRange.
 *
 * With exactly one range per thread, the slowest thread (preempted by the OS, or running on
 * an efficiency core) delays the whole parallel section. Several ranges per thread let the
 * work-stealing scheduler rebalance: a slow thread simply processes fewer ranges. Measured on
 * an Apple M3 Max with 10 threads, a compute-bound loop went from 3.5x to 8.6x speed-up when
 * moving from 1 to 4 ranges per thread. The cost is one extra task per additional range
 * (about a microsecond each), so callers whose per-range work has a fixed cost proportional to
 * the whole problem (e.g. a per-task full-vector reduction) should pass 1.
 */
inline constexpr unsigned int defaultRangesPerThread = 4;

/**
 * Represents an iterable sequence in a container
 */
template<class InputIt>
struct Range
{
    InputIt start;
    InputIt end;

    Range(InputIt s, InputIt e) : start(s), end(e) {}
};

template<class InputIt, class Distance>
void advance(InputIt& it, Distance n)
{
    if constexpr (std::is_integral_v<InputIt>)
    {
        it += n;
    }
    else
    {
        std::advance(it, n);
    }
}

/**
 * Function returning a list of ranges from an iterable container.
 * The number of ranges depends on:
 *  1) the desired number of ranges provided in a parameter
 *  2) the number of elements in the container
 * The number of elements in each range is homogeneous, except for the last range which may contain
 * more elements.
 */
template<class InputIt>
sofa::type::vector<Range<InputIt> >
makeRangesForLoop(const InputIt first, const InputIt last, const unsigned int nbRangesHint)
{
    sofa::type::vector<Range<InputIt> > ranges;

    if (first == last)
    {
        return ranges;
    }

    unsigned int nbElements = 0;
    if constexpr (std::is_integral_v<InputIt>)
    {
        nbElements = static_cast<unsigned int>(last - first);
    }
    else
    {
        nbElements = static_cast<unsigned int>(std::distance(first, last));
    }

    const unsigned int nbRanges = std::min(nbRangesHint, nbElements);
    ranges.reserve(nbRanges);

    const auto nbElementsPerRange = nbElements / nbRanges;

    Range<InputIt> r { first, first};
    sofa::simulation::advance(r.end, nbElementsPerRange);

    for (unsigned int i = 0; i < nbRanges - 1; ++i)
    {
        ranges.emplace_back(r);

        sofa::simulation::advance(r.start, nbElementsPerRange);
        sofa::simulation::advance(r.end, nbElementsPerRange);
    }

    ranges.emplace_back(r.start, last);

    return ranges;
}

/**
 * Applies the given function object f to the result of dereferencing every iterator in the
 * range [first, last), in order.
 */
template<class InputIt, class UnaryFunction>
UnaryFunction forEach(InputIt first, InputIt last, UnaryFunction f)
{
    if constexpr (std::is_integral_v<InputIt>)
    {
        for (; first != last; ++first)
        {
            f(first);
        }
        return f;
    }
    else
    {
        return std::for_each(first, last, f);
    }
}

/**
 * Applies the given function object f to the Range [first, last)
 *
 * The signature of the function f should be equivalent to the following:
 * void fun(const Range<InputIt>& a);
 * The signature does not need to have const &
 */
template<class InputIt, class UnaryFunction>
UnaryFunction forEachRange(InputIt first, InputIt last, UnaryFunction f)
{
    Range<InputIt> r{ first, last};
    f(r);

    return f;
}

namespace detail
{

/**
 * Node of a binary tree of ranges, used as a task by parallelForEachRange.
 *
 * Leaves apply the function object to their range. An internal node covers the union of its
 * children's ranges: when it runs, it queues its right child on the current thread and
 * descends into its left child inline, down to a leaf. This is the divide-and-conquer
 * pattern of Cilk/TBB: the thread that starts the section only queues log2(n) tasks, and every
 * thief that steals a subtree splits it in its own queue, so no single queue becomes a
 * bottleneck when many threads fetch small ranges at once.
 *
 * All nodes are built up-front in a container owned by the caller for the duration of the
 * section: no allocation while the section runs, and the scheduler must not free the tasks
 * (MemoryAlloc::Stack). The function object is referenced, not copied.
 */
template<class InputIt, class UnaryFunction>
class RangeTask final : public Task
{
public:
    RangeTask(TaskScheduler& scheduler, Task::Status& status, const Range<InputIt>& range, UnaryFunction& f)
        : Task(-1)
        , m_scheduler(&scheduler)
        , m_status(&status)
        , m_range(range)
        , m_function(&f)
    {}

    void setChildren(RangeTask* left, RangeTask* right)
    {
        m_left = left;
        m_right = right;
    }

    MemoryAlloc run() final
    {
        RangeTask* node = this;
        while (node->m_left != nullptr)
        {
            m_scheduler->addTask(node->m_right);
            node = node->m_left;
        }
        (*m_function)(node->m_range);
        return MemoryAlloc::Stack;
    }

    Task::Status* getStatus() const final { return m_status; }

private:
    TaskScheduler* m_scheduler;
    Task::Status* m_status;
    Range<InputIt> m_range;
    UnaryFunction* m_function;
    RangeTask* m_left { nullptr };
    RangeTask* m_right { nullptr };
};

/**
 * Builds the tree of RangeTask over the leaf ranges [firstLeaf, lastLeaf) and returns its root.
 * Nodes are appended to @p nodes, which must have enough capacity for 2 * nbLeaves - 1 nodes
 * so that pointers stay valid.
 */
template<class InputIt, class UnaryFunction>
RangeTask<InputIt, UnaryFunction>* buildRangeTree(
    std::vector<RangeTask<InputIt, UnaryFunction>>& nodes,
    const sofa::type::vector<Range<InputIt>>& leaves, const std::size_t firstLeaf, const std::size_t lastLeaf,
    TaskScheduler& scheduler, Task::Status& status, UnaryFunction& f)
{
    if (lastLeaf - firstLeaf == 1)
    {
        nodes.emplace_back(scheduler, status, leaves[firstLeaf], f);
        return &nodes.back();
    }

    const std::size_t middle = firstLeaf + (lastLeaf - firstLeaf) / 2;
    nodes.emplace_back(scheduler, status, Range<InputIt>(leaves[firstLeaf].start, leaves[lastLeaf - 1].end), f);
    RangeTask<InputIt, UnaryFunction>* node = &nodes.back();

    RangeTask<InputIt, UnaryFunction>* left = buildRangeTree(nodes, leaves, firstLeaf, middle, scheduler, status, f);
    RangeTask<InputIt, UnaryFunction>* right = buildRangeTree(nodes, leaves, middle, lastLeaf, scheduler, status, f);
    node->setChildren(left, right);
    return node;
}

}

/**
 * Applies in parallel the given function object f to a list of ranges generated from [first, last)
 *
 * The signature of the function f should be equivalent to the following:
 * void fun(const Range<InputIt>& a);
 * The signature does not need to have const &.
 *
 * A task scheduler must be provided and correctly initialized. The number of generated ranges
 * is the number of threads of the task scheduler multiplied by @p rangesPerThread (clamped to
 * the number of elements). See @ref defaultRangesPerThread for the rationale.
 */
template<class InputIt, class UnaryFunction>
UnaryFunction parallelForEachRange(TaskScheduler& taskScheduler, InputIt first, InputIt last, UnaryFunction f,
                                   const unsigned int rangesPerThread = defaultRangesPerThread)
{
    if (first != last)
    {
        const auto taskSchedulerThreadCount = taskScheduler.getThreadCount();
        if (taskSchedulerThreadCount == 0)
        {
            msg_error("parallelForEach") << "Task scheduler does not appear to be initialized. Cannot perform parallel tasks.";
            return forEachRange(first, last, f);
        }

        const unsigned int nbRanges = taskSchedulerThreadCount * std::max(1u, rangesPerThread);
        const auto ranges = makeRangesForLoop<InputIt>(first, last, nbRanges);

        CpuTaskStatus status;

        // Tree of stack-like tasks over the ranges, allocated at once: no per-task heap
        // allocation nor type erasure. Only the root is queued here; internal nodes queue
        // their right child when they run (see detail::RangeTask).
        std::vector<detail::RangeTask<InputIt, UnaryFunction>> tasks;
        tasks.reserve(2 * ranges.size() - 1);
        auto* root = detail::buildRangeTree(tasks, ranges, 0, ranges.size(), taskScheduler, status, f);

        taskScheduler.addTask(root);
        taskScheduler.workUntilDone(&status);
    }
    return f;
}

/**
 * Applies the given function object f to the result of dereferencing every iterator in the
 * range [first, last), in parallel.
 */
template<class InputIt, class UnaryFunction>
UnaryFunction parallelForEach(TaskScheduler& taskScheduler, InputIt first, InputIt last, UnaryFunction f,
                              const unsigned int rangesPerThread = defaultRangesPerThread)
{
    parallelForEachRange(taskScheduler, first, last,
        [&f](const Range<InputIt>& r)
        {
            forEach(r.start, r.end, f);
        }, rangesPerThread);
    return f;
}


enum class ForEachExecutionPolicy : bool
{
    SEQUENTIAL = false,
    PARALLEL
};

template<class InputIt, class UnaryFunction>
UnaryFunction forEachRange(const ForEachExecutionPolicy execution, TaskScheduler& taskScheduler,
                      InputIt first,
                      InputIt last, UnaryFunction f,
                      const unsigned int rangesPerThread = defaultRangesPerThread)
{
    if (execution == ForEachExecutionPolicy::PARALLEL)
    {
        return parallelForEachRange(taskScheduler, first, last, f, rangesPerThread);
    }
    return forEachRange(first, last, f);
}

template<class InputIt, class UnaryFunction>
UnaryFunction forEach(const ForEachExecutionPolicy execution, TaskScheduler& taskScheduler,
                      InputIt first,
                      InputIt last, UnaryFunction f,
                      const unsigned int rangesPerThread = defaultRangesPerThread)
{
    if (execution == ForEachExecutionPolicy::PARALLEL)
    {
        return parallelForEach(taskScheduler, first, last, f, rangesPerThread);
    }
    return forEach(first, last, f);
}

}
