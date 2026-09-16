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
#include <sofa/helper/system/Cpu.h>

#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#include <vector>
#elif defined(__linux__)
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#endif

namespace sofa::helper::system
{

namespace
{

#if defined(__APPLE__)

unsigned querySysctlUnsigned(const char* name)
{
    int value = 0;
    std::size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value > 0)
    {
        return static_cast<unsigned>(value);
    }
    return 0;
}

#elif defined(__linux__)

/// Physical core identifier: core ids are only unique within a package.
using CoreKey = std::pair<int, int>;

/// Maps every physical core to its capacity as reported by the kernel
/// (/sys/devices/system/cpu/cpuN/cpu_capacity, -1 when the file is absent).
/// Returns an empty map when the sysfs topology is unavailable.
std::map<CoreKey, int> readCoreTopology()
{
    namespace fs = std::filesystem;
    std::map<CoreKey, int> cores;

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator("/sys/devices/system/cpu", ec))
    {
        const std::string name = entry.path().filename().string();
        if (name.size() < 4 || name.compare(0, 3, "cpu") != 0 || !std::isdigit(static_cast<unsigned char>(name[3])))
        {
            continue;
        }

        const auto readInt = [](const fs::path& file)
        {
            std::ifstream in(file);
            int value = -1;
            in >> value;
            return value;
        };

        const int coreId = readInt(entry.path() / "topology" / "core_id");
        if (coreId < 0)
        {
            continue;
        }
        const int packageId = readInt(entry.path() / "topology" / "physical_package_id");
        const int capacity = readInt(entry.path() / "cpu_capacity");

        auto& stored = cores[{packageId, coreId}];
        stored = std::max(stored, capacity);
    }

    if (ec)
    {
        cores.clear();
    }
    return cores;
}

#endif

/// Platform-specific query. Returns 0 when the information is unavailable.
unsigned queryPhysicalCoreCount()
{
#if defined(__APPLE__)
    return querySysctlUnsigned("hw.physicalcpu");

#elif defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0)
    {
        return 0;
    }

    std::vector<char> buffer(length);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length))
    {
        return 0;
    }

    unsigned count = 0;
    for (DWORD offset = 0; offset < length;)
    {
        const auto* current = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (current->Relationship == RelationProcessorCore)
        {
            ++count;
        }
        offset += current->Size;
    }
    return count;

#elif defined(__linux__)
    return static_cast<unsigned>(readCoreTopology().size());

#else
    return 0;
#endif
}

/// Platform-specific query. Returns 0 when the information is unavailable.
unsigned queryPerformanceCoreCount()
{
#if defined(__APPLE__)
    // perflevel0 is always the highest-performance core type.
    return querySysctlUnsigned("hw.perflevel0.physicalcpu");

#elif defined(__linux__)
    const auto cores = readCoreTopology();
    if (cores.empty())
    {
        return 0;
    }

    int maxCapacity = -1;
    for (const auto& [key, capacity] : cores)
    {
        maxCapacity = std::max(maxCapacity, capacity);
    }
    if (maxCapacity < 0)
    {
        // cpu_capacity is not exposed: assume homogeneous cores
        return static_cast<unsigned>(cores.size());
    }

    unsigned count = 0;
    for (const auto& [key, capacity] : cores)
    {
        if (capacity == maxCapacity)
        {
            ++count;
        }
    }
    return count;

#else
    // No portable way to distinguish core types: assume homogeneous cores
    return 0;
#endif
}

} // namespace

unsigned getLogicalCoreCount()
{
    const unsigned logical = std::thread::hardware_concurrency();
    return logical == 0 ? 1u : logical;
}

unsigned getPhysicalCoreCount()
{
    const unsigned logical = getLogicalCoreCount();
    const unsigned physical = queryPhysicalCoreCount();

    if (physical == 0 || physical > logical)
    {
        return logical;
    }
    return physical;
}

unsigned getPerformanceCoreCount()
{
    const unsigned physical = getPhysicalCoreCount();
    const unsigned performance = queryPerformanceCoreCount();

    if (performance == 0 || performance > physical)
    {
        return physical;
    }
    return performance;
}

} // namespace sofa::helper::system
