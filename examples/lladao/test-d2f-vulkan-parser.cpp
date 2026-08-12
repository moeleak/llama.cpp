#include "ggml-vulkan-d2f.h"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct packed_graph {
    ggml_context *             ctx   = nullptr;
    ggml_cgraph *              graph = nullptr;
    std::vector<ggml_tensor *> lanes;
    std::vector<ggml_tensor *> concats;
    ggml_tensor *              join = nullptr;

    explicit packed_graph(int masked_lane = -1, bool balanced = false) {
        ggml_init_params params = {
            16 * 1024 * 1024,
            nullptr,
            true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context");
        }
        graph = ggml_new_graph_custom(ctx, 128, false);

        const std::vector<int64_t> lengths =
            balanced ? std::vector<int64_t>{ 2, 5, 3, 4 } : std::vector<int64_t>{ 2, 5, 3 };
        lanes.reserve(lengths.size());
        for (size_t lane = 0; lane < lengths.size(); ++lane) {
            const int64_t length = lengths[lane];
            ggml_tensor * q      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, length, 2, 1);
            ggml_tensor * k      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, length, 2, 1);
            ggml_tensor * v      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, length, 2, 1);
            ggml_tensor * mask   = nullptr;
            if (static_cast<int>(lane) == masked_lane) {
                mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, length, length, 1, 1);
            }
            ggml_tensor * flash = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
            ggml_tensor * output =
                ggml_reshape_2d(ctx, flash, flash->ne[0] * flash->ne[1], flash->ne[2] * flash->ne[3]);
            ggml_format_name(output, "d2f_packed_attn_lane-7-%d", static_cast<int>(lane));
            lanes.push_back(output);
        }

        ggml_tensor * left = ggml_concat(ctx, lanes[0], lanes[1], 1);
        concats.push_back(left);
        if (balanced) {
            ggml_tensor * right = ggml_concat(ctx, lanes[2], lanes[3], 1);
            concats.push_back(right);
            join = ggml_concat(ctx, left, right, 1);
        } else {
            join = ggml_concat(ctx, left, lanes[2], 1);
        }
        concats.push_back(join);
        ggml_set_name(join, "d2f_packed_attn_join-7");
        ggml_build_forward_expand(graph, join);
    }

    ~packed_graph() { ggml_free(ctx); }

    packed_graph(const packed_graph &)             = delete;
    packed_graph & operator=(const packed_graph &) = delete;
};

void require_rejected(const ggml_vk_d2f::packed_attention_parse_result & result) {
    if (result.status != ggml_vk_d2f::packed_attention_parse_status::rejected || !result.regions.empty() ||
        result.reason.empty()) {
        throw std::runtime_error("packed graph was not rejected fail-closed");
    }
}

void test_valid_unequal_lanes() {
    packed_graph fixture;
    const auto   result = ggml_vk_d2f::parse_packed_attention_graph(fixture.graph);
    if (result.status != ggml_vk_d2f::packed_attention_parse_status::accepted || result.regions.size() != 1 ||
        result.regions[0].layer != 7 || result.regions[0].join != fixture.join || result.regions[0].lanes.size() != 3 ||
        result.regions[0].concat_postorder.size() != 2 || result.regions[0].concat_postorder.back() != fixture.join) {
        throw std::runtime_error("valid three-lane packed graph was not accepted");
    }

    const std::array<int64_t, 3> expected_lengths = { 2, 5, 3 };
    for (size_t lane = 0; lane < result.regions[0].lanes.size(); ++lane) {
        const auto & parsed = result.regions[0].lanes[lane];
        if (parsed.id != static_cast<int>(lane) || parsed.output != fixture.lanes[lane] ||
            parsed.flash_attn == nullptr || parsed.flash_attn->op != GGML_OP_FLASH_ATTN_EXT ||
            parsed.flash_attn->src[3] != nullptr || parsed.output->ne[1] != expected_lengths[lane] ||
            parsed.flash_node_index < result.regions[0].region_start ||
            parsed.flash_node_index > parsed.output_node_index ||
            parsed.output_node_index >= result.regions[0].join_node_index) {
            throw std::runtime_error("valid packed lane metadata is invalid");
        }
    }
}

void test_missing_join() {
    packed_graph fixture;
    ggml_set_name(fixture.join, "plain_join");
    require_rejected(ggml_vk_d2f::parse_packed_attention_graph(fixture.graph));
}

void test_lane_gap_and_duplicate() {
    {
        packed_graph fixture;
        ggml_set_name(fixture.lanes[2], "d2f_packed_attn_lane-7-3");
        require_rejected(ggml_vk_d2f::parse_packed_attention_graph(fixture.graph));
    }
    {
        packed_graph fixture;
        ggml_set_name(fixture.lanes[2], "d2f_packed_attn_lane-7-1");
        require_rejected(ggml_vk_d2f::parse_packed_attention_graph(fixture.graph));
    }
}

void test_masked_flash_attention() {
    packed_graph fixture(1);
    require_rejected(ggml_vk_d2f::parse_packed_attention_graph(fixture.graph));
}

void test_extra_consumer() {
    packed_graph  fixture;
    ggml_tensor * extra =
        ggml_reshape_2d(fixture.ctx, fixture.lanes[0], fixture.lanes[0]->ne[0], fixture.lanes[0]->ne[1]);
    ggml_build_forward_expand(fixture.graph, extra);
    require_rejected(ggml_vk_d2f::parse_packed_attention_graph(fixture.graph));
}

void test_cross_lane_dependency() {
    packed_graph fixture;
    fixture.lanes[1]->src[1] = fixture.lanes[0];
    require_rejected(ggml_vk_d2f::parse_packed_attention_graph(fixture.graph));
}

void test_wrong_concat() {
    packed_graph fixture;
    ggml_set_op_params_i32(fixture.join, 0, 0);
    require_rejected(ggml_vk_d2f::parse_packed_attention_graph(fixture.graph));
}

void test_malformed_marker() {
    packed_graph fixture;
    ggml_set_name(fixture.join, "d2f_packed_attn_join-7-extra");
    require_rejected(ggml_vk_d2f::parse_packed_attention_graph(fixture.graph));
}

int find_node_index(ggml_cgraph * graph, const ggml_tensor * needle) {
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        if (ggml_graph_node(graph, index) == needle) {
            return index;
        }
    }
    throw std::runtime_error("test tensor is missing from graph");
}

void move_node_before(ggml_cgraph * graph, const ggml_tensor * node, const ggml_tensor * before) {
    ggml_tensor ** nodes        = ggml_graph_nodes(graph);
    const int      node_index   = find_node_index(graph, node);
    const int      before_index = find_node_index(graph, before);
    if (node_index <= before_index) {
        throw std::runtime_error("test node is not after its requested destination");
    }
    ggml_tensor * moved = nodes[node_index];
    for (int index = node_index; index > before_index; --index) {
        nodes[index] = nodes[index - 1];
    }
    nodes[before_index] = moved;
}

ggml_tensor * add_final_residual_rows(packed_graph & fixture) {
    ggml_tensor * residual = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F32, fixture.join->ne[0], fixture.join->ne[1]);
    ggml_tensor * ids      = ggml_new_tensor_1d(fixture.ctx, GGML_TYPE_I32, fixture.join->ne[1]);
    ggml_tensor * rows     = ggml_get_rows(fixture.ctx, residual, ids);
    ggml_tensor * sum      = ggml_add(fixture.ctx, fixture.join, rows);
    ggml_build_forward_expand(fixture.graph, sum);
    return rows;
}

void test_preexpanded_residual_get_rows() {
    packed_graph  fixture;
    ggml_tensor * rows = add_final_residual_rows(fixture);
    move_node_before(fixture.graph, rows, fixture.lanes[0]->src[0]);

    const auto result = ggml_vk_d2f::parse_packed_attention_graph(fixture.graph);
    if (result.status != ggml_vk_d2f::packed_attention_parse_status::accepted || result.regions.size() != 1 ||
        find_node_index(fixture.graph, rows) >= result.regions[0].region_start) {
        throw std::runtime_error("pre-expanded final residual GET_ROWS was not kept outside the packed region");
    }
}

void test_residual_get_rows_inside_region() {
    packed_graph  fixture;
    ggml_tensor * rows = add_final_residual_rows(fixture);
    move_node_before(fixture.graph, rows, fixture.join);

    const auto result = ggml_vk_d2f::parse_packed_attention_graph(fixture.graph);
    require_rejected(result);
    if (result.reason.find("GET_ROWS") == std::string::npos) {
        throw std::runtime_error("foreign GET_ROWS rejection did not identify the unsafe op");
    }
}

void test_foreign_compute_inside_region() {
    packed_graph  fixture;
    ggml_tensor * input   = ggml_new_tensor_1d(fixture.ctx, GGML_TYPE_F32, 8);
    ggml_tensor * foreign = ggml_sqr(fixture.ctx, input);
    ggml_graph_add_node(fixture.graph, foreign);

    ggml_tensor ** nodes      = ggml_graph_nodes(fixture.graph);
    const int      join_index = find_node_index(fixture.graph, fixture.join);
    const int      last_index = ggml_graph_n_nodes(fixture.graph) - 1;
    nodes[last_index]         = nodes[join_index];
    nodes[join_index]         = foreign;
    require_rejected(ggml_vk_d2f::parse_packed_attention_graph(fixture.graph));
}

void test_concat_replay_follows_graph_order() {
    packed_graph   fixture(-1, true);
    ggml_tensor ** nodes       = ggml_graph_nodes(fixture.graph);
    const int      left_index  = find_node_index(fixture.graph, fixture.concats[0]);
    const int      right_index = find_node_index(fixture.graph, fixture.concats[1]);
    ggml_tensor *  left        = nodes[left_index];
    for (int index = left_index; index < right_index; ++index) {
        nodes[index] = nodes[index + 1];
    }
    nodes[right_index] = left;

    const auto result = ggml_vk_d2f::parse_packed_attention_graph(fixture.graph);
    if (result.status != ggml_vk_d2f::packed_attention_parse_status::accepted || result.regions.size() != 1 ||
        result.regions[0].concat_postorder.size() != 3 || result.regions[0].concat_postorder[0] != fixture.concats[1] ||
        result.regions[0].concat_postorder[1] != fixture.concats[0] ||
        result.regions[0].concat_postorder[2] != fixture.join) {
        throw std::runtime_error("concat replay order does not match graph order");
    }
}

void test_compute_queue_planning() {
    const auto single            = ggml_vk_d2f::plan_compute_queues(3, 1, true, false);
    const auto parallel          = ggml_vk_d2f::plan_compute_queues(3, 3, true, false);
    const auto reserved          = ggml_vk_d2f::plan_compute_queues(3, 3, true, true);
    const auto distinct_transfer = ggml_vk_d2f::plan_compute_queues(3, 2, false, true);
    const auto no_queues         = ggml_vk_d2f::plan_compute_queues(3, 0, true, true);

    if (single.compute_queue_count != 1 || single.reserve_transfer_queue || parallel.compute_queue_count != 3 ||
        parallel.reserve_transfer_queue || reserved.compute_queue_count != 2 || !reserved.reserve_transfer_queue ||
        distinct_transfer.compute_queue_count != 2 || distinct_transfer.reserve_transfer_queue ||
        no_queues.compute_queue_count != 0 || no_queues.reserve_transfer_queue) {
        throw std::runtime_error("Vulkan compute queue planning is invalid");
    }
}

}  // namespace

int main() {
    test_valid_unequal_lanes();
    test_missing_join();
    test_lane_gap_and_duplicate();
    test_masked_flash_attention();
    test_extra_consumer();
    test_cross_lane_dependency();
    test_wrong_concat();
    test_malformed_marker();
    test_preexpanded_residual_get_rows();
    test_residual_get_rows_inside_region();
    test_foreign_compute_inside_region();
    test_concat_replay_follows_graph_order();
    test_compute_queue_planning();
    return 0;
}
