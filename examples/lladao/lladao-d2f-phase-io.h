#pragma once

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace lladao {

// Per-process counters sampled at phase boundaries. These counters answer
// whether a slow phase spent time executing on the CPU, faulting pages, or
// doing storage I/O. They intentionally do not claim to measure GPU work.
struct d2f_phase_resource_usage {
    double process_cpu_seconds = 0.0;
    uint64_t minor_page_faults = 0;
    uint64_t major_page_faults = 0;
    uint64_t block_input_operations = 0;
    uint64_t block_output_operations = 0;
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
    bool proc_io_available = false;
};

namespace detail {

struct d2f_proc_io_counters {
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
    bool available = false;
};

struct d2f_process_resource_snapshot {
    double process_cpu_seconds = 0.0;
    uint64_t minor_page_faults = 0;
    uint64_t major_page_faults = 0;
    uint64_t block_input_operations = 0;
    uint64_t block_output_operations = 0;
    d2f_proc_io_counters proc_io;
};

inline bool parse_u64_decimal(std::string_view text, uint64_t & value) {
    const size_t first = text.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
        return false;
    }
    const size_t last = text.find_last_not_of(" \t\r");
    uint64_t parsed = 0;
    for (size_t i = first; i <= last; ++i) {
        const char c = text[i];
        if (c < '0' || c > '9') {
            return false;
        }
        const uint64_t digit = static_cast<uint64_t>(c - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return false;
        }
        parsed = parsed * 10 + digit;
    }
    value = parsed;
    return true;
}

inline d2f_proc_io_counters parse_proc_self_io(std::string_view contents) {
    d2f_proc_io_counters result;
    bool have_read = false;
    bool have_write = false;
    size_t begin = 0;
    while (begin < contents.size()) {
        const size_t newline = contents.find('\n', begin);
        const size_t end = newline == std::string_view::npos ? contents.size() : newline;
        const std::string_view line = contents.substr(begin, end - begin);
        const size_t colon = line.find(':');
        if (colon != std::string_view::npos) {
            const std::string_view key = line.substr(0, colon);
            uint64_t value = 0;
            if (key == "read_bytes" && parse_u64_decimal(line.substr(colon + 1), value)) {
                result.read_bytes = value;
                have_read = true;
            } else if (key == "write_bytes" && parse_u64_decimal(line.substr(colon + 1), value)) {
                result.write_bytes = value;
                have_write = true;
            }
        }
        if (newline == std::string_view::npos) {
            break;
        }
        begin = newline + 1;
    }
    result.available = have_read && have_write;
    return result;
}

inline d2f_proc_io_counters read_proc_self_io() {
#if defined(__linux__) || defined(__ANDROID__)
    std::ifstream file("/proc/self/io", std::ios::binary);
    if (!file) {
        return {};
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    if (!file.good() && !file.eof()) {
        return {};
    }
    return parse_proc_self_io(contents.str());
#else
    return {};
#endif
}

inline uint64_t nonnegative_counter(long value) {
    return value > 0 ? static_cast<uint64_t>(value) : 0;
}

inline double timeval_seconds(long seconds, long microseconds) {
    return static_cast<double>(seconds) + static_cast<double>(microseconds) / 1.0e6;
}

inline d2f_process_resource_snapshot snapshot_process_resources() {
    d2f_process_resource_snapshot result;
#if !defined(_WIN32)
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        result.process_cpu_seconds =
                timeval_seconds(usage.ru_utime.tv_sec, usage.ru_utime.tv_usec) +
                timeval_seconds(usage.ru_stime.tv_sec, usage.ru_stime.tv_usec);
        result.minor_page_faults = nonnegative_counter(usage.ru_minflt);
        result.major_page_faults = nonnegative_counter(usage.ru_majflt);
        result.block_input_operations = nonnegative_counter(usage.ru_inblock);
        result.block_output_operations = nonnegative_counter(usage.ru_oublock);
    }
#else
    const std::clock_t cpu = std::clock();
    if (cpu != static_cast<std::clock_t>(-1)) {
        result.process_cpu_seconds = static_cast<double>(cpu) / CLOCKS_PER_SEC;
    }
#endif
    result.proc_io = read_proc_self_io();
    return result;
}

inline uint64_t monotonic_delta(uint64_t before, uint64_t after) {
    return after >= before ? after - before : 0;
}

inline d2f_phase_resource_usage resource_usage_delta(
        const d2f_process_resource_snapshot & before,
        const d2f_process_resource_snapshot & after) {
    d2f_phase_resource_usage result;
    result.process_cpu_seconds = std::max(0.0, after.process_cpu_seconds - before.process_cpu_seconds);
    result.minor_page_faults = monotonic_delta(before.minor_page_faults, after.minor_page_faults);
    result.major_page_faults = monotonic_delta(before.major_page_faults, after.major_page_faults);
    result.block_input_operations = monotonic_delta(
            before.block_input_operations, after.block_input_operations);
    result.block_output_operations = monotonic_delta(
            before.block_output_operations, after.block_output_operations);
    result.proc_io_available = before.proc_io.available && after.proc_io.available;
    if (result.proc_io_available) {
        result.read_bytes = monotonic_delta(before.proc_io.read_bytes, after.proc_io.read_bytes);
        result.write_bytes = monotonic_delta(before.proc_io.write_bytes, after.proc_io.write_bytes);
    }
    return result;
}

} // namespace detail
} // namespace lladao
