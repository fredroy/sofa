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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(_M_ARM64)
#include <intrin.h>
#endif

namespace sofa::simulation
{

/// Hint to the CPU that the current thread is in a spin-wait loop
inline void cpuRelax() noexcept
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif defined(_M_ARM64)
    __yield();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#endif
}

/**
 * Test-and-test-and-set spin lock.
 *
 * Waiting threads spin on a plain load and only attempt the atomic exchange once the lock
 * looks free, so contended waiting does not keep invalidating the owner's cache line.
 * The lock occupies its own cache line to avoid false sharing with neighbouring members.
 */
class alignas(64) SpinLock
{
public:

    SpinLock() = default;

    SpinLock(const SpinLock&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;

    bool try_lock()
    {
        return !m_locked.load(std::memory_order_relaxed)
            && !m_locked.exchange(true, std::memory_order_acquire);
    }

    void lock()
    {
        for (;;)
        {
            if (!m_locked.exchange(true, std::memory_order_acquire))
            {
                return;
            }
            while (m_locked.load(std::memory_order_relaxed))
            {
                cpuRelax();
            }
        }
    }

    void unlock()
    {
        m_locked.store(false, std::memory_order_release);
    }

private:

    std::atomic<bool> m_locked { false };
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
