#include "lladao-d2f-phase-io.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    using namespace lladao::detail;

    const d2f_proc_io_counters parsed = parse_proc_self_io(
            "rchar: 99\n"
            "wchar: 17\n"
            "syscr: 3\n"
            "read_bytes: 4096\n"
            "write_bytes: 8192\n"
            "cancelled_write_bytes: 0\n");
    require(parsed.available && parsed.read_bytes == 4096 && parsed.write_bytes == 8192,
            "/proc/self/io parser did not read storage byte counters");
    require(!parse_proc_self_io("read_bytes: 1\nwrite_bytes: invalid\n").available,
            "/proc/self/io parser accepted an incomplete snapshot");
    require(!parse_proc_self_io("read_bytes: 18446744073709551616\nwrite_bytes: 1\n").available,
            "/proc/self/io parser accepted an overflowing counter");

    d2f_process_resource_snapshot before;
    before.process_cpu_seconds = 3.25;
    before.minor_page_faults = 10;
    before.major_page_faults = 2;
    before.block_input_operations = 4;
    before.block_output_operations = 5;
    before.proc_io = { 1024, 2048, true };

    d2f_process_resource_snapshot after;
    after.process_cpu_seconds = 4.0;
    after.minor_page_faults = 18;
    after.major_page_faults = 3;
    after.block_input_operations = 7;
    after.block_output_operations = 11;
    after.proc_io = { 5120, 10240, true };

    const lladao::d2f_phase_resource_usage delta = resource_usage_delta(before, after);
    require(std::abs(delta.process_cpu_seconds - 0.75) < 1e-9,
            "process CPU delta is incorrect");
    require(delta.minor_page_faults == 8 && delta.major_page_faults == 1 &&
                    delta.block_input_operations == 3 && delta.block_output_operations == 6,
            "rusage counter delta is incorrect");
    require(delta.proc_io_available && delta.read_bytes == 4096 && delta.write_bytes == 8192,
            "storage I/O byte delta is incorrect");

    after.process_cpu_seconds = 2.0;
    after.minor_page_faults = 1;
    after.proc_io.available = false;
    const lladao::d2f_phase_resource_usage reset = resource_usage_delta(before, after);
    require(reset.process_cpu_seconds == 0.0 && reset.minor_page_faults == 0 &&
                    !reset.proc_io_available && reset.read_bytes == 0 && reset.write_bytes == 0,
            "counter reset or unavailable proc I/O did not fail closed");

    const d2f_process_resource_snapshot live_before = snapshot_process_resources();
    volatile uint64_t accumulator = 0;
    for (uint64_t i = 0; i < 10000; ++i) {
        accumulator += i;
    }
    (void) accumulator;
    const d2f_process_resource_snapshot live_after = snapshot_process_resources();
    const lladao::d2f_phase_resource_usage live = resource_usage_delta(live_before, live_after);
    require(live.process_cpu_seconds >= 0.0, "live process snapshot produced negative CPU time");

    return 0;
}
