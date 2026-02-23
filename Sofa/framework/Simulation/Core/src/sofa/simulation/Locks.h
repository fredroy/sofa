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

#include <thread>
#include <mutex>
#include <atomic>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace sofa::simulation
{

/// Emit a CPU-level hint that this thread is in a spin-wait loop.
/// Reduces power consumption and avoids starving the other hyperthread
/// on the same physical core.
inline void spinLoopPause()
{
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

class SpinLock
{
public:

    SpinLock()
    : m_locked(false)
    {}

    ~SpinLock()
    {
        unlock();
    }

    bool try_lock()
    {
        // TTAS: read first to avoid unnecessary cache-line invalidation
        return !m_locked.load(std::memory_order_relaxed) &&
               !m_locked.exchange(true, std::memory_order_acquire);
    }

    void lock()
    {
        // Fast path: try to acquire immediately
        if (!m_locked.exchange(true, std::memory_order_acquire))
            return;

        // Slow path: TTAS (Test-and-Test-And-Set).
        // Spin on a relaxed load (read-only) to avoid bouncing the
        // cache line between cores on every iteration.  Only attempt
        // the expensive exchange when the lock appears free.
        for (;;)
        {
            while (m_locked.load(std::memory_order_relaxed))
            {
                spinLoopPause();
            }
            if (!m_locked.exchange(true, std::memory_order_acquire))
                return;
        }
    }

    void unlock()
    {
        m_locked.store(false, std::memory_order_release);
    }

private:

    std::atomic<bool> m_locked;
};
        
        
        
class ScopedLock
{
public:
            
    explicit ScopedLock( SpinLock & lock ): m_spinlock( lock )
    {
        m_spinlock.lock();
    }
            
    ~ScopedLock()
    {
        m_spinlock.unlock();
    }
            
    ScopedLock( ScopedLock const & ) = delete;
    ScopedLock & operator=( ScopedLock const & ) = delete;
            
private:
            
    SpinLock& m_spinlock;
};

} // namespace sofa::simulation
