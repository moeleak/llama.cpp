#pragma once

#include "ggml-impl.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ggml_vk_d2f {

enum class packed_attention_parse_status {
    absent,
    accepted,
    rejected,
};

struct compute_queue_plan {
    uint32_t compute_queue_count    = 0;
    bool     reserve_transfer_queue = false;
};

inline compute_queue_plan plan_compute_queues(uint32_t requested_compute_queues,
                                              uint32_t available_queues,
                                              bool     same_family_transfer,
                                              bool     async_transfer) {
    if (available_queues == 0) {
        return {};
    }

    uint32_t compute_queue_count = std::min(std::max(requested_compute_queues, 1u), available_queues);
    if (same_family_transfer && async_transfer && compute_queue_count == available_queues && compute_queue_count > 1) {
        --compute_queue_count;
    }

    return {
        compute_queue_count,
        same_family_transfer && async_transfer && compute_queue_count < available_queues,
    };
}

struct packed_attention_lane {
    int                              id         = -1;
    const ggml_tensor *              output     = nullptr;
    const ggml_tensor *              flash_attn = nullptr;
    std::vector<const ggml_tensor *> branch;
    int                              flash_node_index  = -1;
    int                              output_node_index = -1;
};

struct packed_attention_region {
    int                                layer = -1;
    const ggml_tensor *                join  = nullptr;
    std::vector<packed_attention_lane> lanes;
    std::vector<const ggml_tensor *>   concat_postorder;
    int                                region_start    = -1;
    int                                join_node_index = -1;
};

struct packed_attention_parse_result {
    packed_attention_parse_status        status = packed_attention_parse_status::absent;
    std::vector<packed_attention_region> regions;
    int                                  rejected_layer = -1;
    std::string                          reason;
};

namespace detail {

static constexpr const char * packed_marker_prefix = "d2f_packed_attn_";
static constexpr const char * packed_lane_prefix   = "d2f_packed_attn_lane-";
static constexpr const char * packed_join_prefix   = "d2f_packed_attn_join-";

inline bool starts_with(const char * value, const char * prefix) {
    return std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

inline bool parse_nonnegative_int(const char *& cursor, int & value) {
    if (*cursor < '0' || *cursor > '9') {
        return false;
    }

    int parsed = 0;
    do {
        const int digit = *cursor - '0';
        if (parsed > (INT_MAX - digit) / 10) {
            return false;
        }
        parsed = parsed * 10 + digit;
        ++cursor;
    } while (*cursor >= '0' && *cursor <= '9');

    value = parsed;
    return true;
}

inline bool parse_lane_name(const char * name, int & layer, int & lane) {
    if (!starts_with(name, packed_lane_prefix)) {
        return false;
    }

    const char * cursor = name + std::strlen(packed_lane_prefix);
    if (!parse_nonnegative_int(cursor, layer) || *cursor++ != '-' || !parse_nonnegative_int(cursor, lane) ||
        *cursor != '\0') {
        return false;
    }
    return true;
}

inline bool parse_join_name(const char * name, int & layer) {
    if (!starts_with(name, packed_join_prefix)) {
        return false;
    }

    const char * cursor = name + std::strlen(packed_join_prefix);
    return parse_nonnegative_int(cursor, layer) && *cursor == '\0';
}

inline bool is_view_or_noop(const ggml_tensor * tensor) {
    return ggml_is_empty(tensor) || tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_TRANSPOSE ||
           tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_NONE;
}

inline bool trace_lane(const ggml_tensor * output, packed_attention_lane & lane) {
    std::vector<const ggml_tensor *> tail;
    const ggml_tensor *              node = output;
    while (node != nullptr && node->op != GGML_OP_FLASH_ATTN_EXT) {
        if (!is_view_or_noop(node) || node->src[0] == nullptr) {
            return false;
        }
        for (int src = 1; src < GGML_MAX_SRC; ++src) {
            if (node->src[src] != nullptr) {
                return false;
            }
        }
        tail.push_back(node);
        node = node->src[0];
    }
    if (node == nullptr || node->src[3] != nullptr) {
        return false;
    }

    lane.flash_attn = node;
    lane.branch.push_back(node);
    for (auto it = tail.rbegin(); it != tail.rend(); ++it) {
        lane.branch.push_back(*it);
    }
    return true;
}

inline bool collect_concat_tree(const ggml_tensor *                   node,
                                const std::set<const ggml_tensor *> & lane_outputs,
                                std::set<const ggml_tensor *> &       seen_lanes,
                                std::vector<const ggml_tensor *> &    concat_postorder) {
    if (lane_outputs.count(node) != 0) {
        return seen_lanes.insert(node).second;
    }
    if (node == nullptr || node->op != GGML_OP_CONCAT || ggml_get_op_params_i32(node, 0) != 1 ||
        node->src[0] == nullptr || node->src[1] == nullptr) {
        return false;
    }
    for (int src = 2; src < GGML_MAX_SRC; ++src) {
        if (node->src[src] != nullptr) {
            return false;
        }
    }
    if (!collect_concat_tree(node->src[0], lane_outputs, seen_lanes, concat_postorder) ||
        !collect_concat_tree(node->src[1], lane_outputs, seen_lanes, concat_postorder)) {
        return false;
    }
    concat_postorder.push_back(node);
    return true;
}

struct packed_group {
    const ggml_tensor *                              join = nullptr;
    std::vector<std::pair<int, const ggml_tensor *>> lanes;
    bool                                             duplicate_join = false;
};

inline packed_attention_parse_result reject(int layer, std::string reason) {
    packed_attention_parse_result result;
    result.status         = packed_attention_parse_status::rejected;
    result.rejected_layer = layer;
    result.reason         = std::move(reason);
    return result;
}

}  // namespace detail

inline packed_attention_parse_result parse_packed_attention_graph(ggml_cgraph * graph) {
    if (graph == nullptr) {
        return detail::reject(-1, "graph is null");
    }

    std::map<int, detail::packed_group>          groups;
    std::unordered_map<const ggml_tensor *, int> node_indices;
    const int                                    node_count = ggml_graph_n_nodes(graph);
    node_indices.reserve(static_cast<size_t>(node_count));

    for (int index = 0; index < node_count; ++index) {
        const ggml_tensor * node = ggml_graph_node(graph, index);
        if (node == nullptr || !node_indices.emplace(node, index).second) {
            return detail::reject(-1, "graph contains a null or duplicate node");
        }

        int layer = -1;
        int lane  = -1;
        if (detail::parse_lane_name(node->name, layer, lane)) {
            groups[layer].lanes.emplace_back(lane, node);
        } else if (detail::parse_join_name(node->name, layer)) {
            detail::packed_group & group = groups[layer];
            if (group.join != nullptr) {
                group.duplicate_join = true;
            } else {
                group.join = node;
            }
        } else if (detail::starts_with(node->name, detail::packed_marker_prefix)) {
            return detail::reject(-1, "graph contains a malformed packed marker");
        }
    }

    if (groups.empty()) {
        return {};
    }

    packed_attention_parse_result result;
    result.status = packed_attention_parse_status::accepted;
    result.regions.reserve(groups.size());

    for (auto & entry : groups) {
        const int              layer = entry.first;
        detail::packed_group & group = entry.second;
        if (group.duplicate_join || group.join == nullptr || group.lanes.size() < 2) {
            return detail::reject(layer, "incomplete or duplicate lane/join markers");
        }

        std::sort(group.lanes.begin(), group.lanes.end(),
                  [](const auto & lhs, const auto & rhs) { return lhs.first < rhs.first; });
        for (size_t lane = 0; lane < group.lanes.size(); ++lane) {
            if (group.lanes[lane].first != static_cast<int>(lane)) {
                return detail::reject(layer, "lane ids are not unique and contiguous from zero");
            }
        }

        packed_attention_region region;
        region.layer           = layer;
        region.join            = group.join;
        region.join_node_index = node_indices.at(group.join);
        region.region_start    = region.join_node_index;
        region.lanes.reserve(group.lanes.size());

        std::set<const ggml_tensor *> lane_outputs;
        for (const auto & marker : group.lanes) {
            lane_outputs.insert(marker.second);
        }

        std::set<const ggml_tensor *> seen_lanes;
        if (!detail::collect_concat_tree(group.join, lane_outputs, seen_lanes, region.concat_postorder) ||
            seen_lanes != lane_outputs || region.concat_postorder.empty() ||
            region.concat_postorder.back() != group.join) {
            return detail::reject(layer, "join is not an exact dim-1 concat tree over all lanes");
        }

        std::set<const ggml_tensor *>                                branch_members;
        std::set<const ggml_tensor *>                                flash_nodes;
        std::unordered_map<const ggml_tensor *, const ggml_tensor *> expected_consumers;
        for (const auto & marker : group.lanes) {
            packed_attention_lane lane;
            lane.id     = marker.first;
            lane.output = marker.second;
            if (!detail::trace_lane(marker.second, lane) || !flash_nodes.insert(lane.flash_attn).second) {
                return detail::reject(layer, "lane does not trace through a unique unmasked FA node");
            }
            int previous_node_index = -1;
            for (size_t branch_index = 0; branch_index < lane.branch.size(); ++branch_index) {
                const ggml_tensor * node    = lane.branch[branch_index];
                const auto          node_it = node_indices.find(node);
                if (node_it == node_indices.end() || node_it->second >= region.join_node_index ||
                    node_it->second <= previous_node_index || !branch_members.insert(node).second) {
                    return detail::reject(layer, "lane node is duplicated or lies outside its join region");
                }
                previous_node_index = node_it->second;
                region.region_start = std::min(region.region_start, node_it->second);
                if (branch_index + 1 < lane.branch.size()) {
                    expected_consumers[node] = lane.branch[branch_index + 1];
                }
            }
            lane.flash_node_index  = node_indices.at(lane.flash_attn);
            lane.output_node_index = node_indices.at(lane.output);
            region.lanes.push_back(std::move(lane));
        }

        std::set<const ggml_tensor *> concat_members(region.concat_postorder.begin(), region.concat_postorder.end());
        for (const ggml_tensor * concat : region.concat_postorder) {
            const auto node_it = node_indices.find(concat);
            if (node_it == node_indices.end() || node_it->second > region.join_node_index) {
                return detail::reject(layer, "concat node lies outside its join region");
            }
            region.region_start = std::min(region.region_start, node_it->second);
            for (int src = 0; src < 2; ++src) {
                const ggml_tensor * child = concat->src[src];
                if (branch_members.count(child) != 0 || concat_members.count(child) != 0) {
                    const auto child_it = node_indices.find(child);
                    if (child_it == node_indices.end() || child_it->second >= node_it->second) {
                        return detail::reject(layer, "concat tree is not topologically ordered");
                    }
                    if (!expected_consumers.emplace(child, concat).second) {
                        return detail::reject(layer, "lane or concat node has multiple tree consumers");
                    }
                }
            }
        }

        // Lane input views can be interleaved with the Flash Attention nodes in
        // the original graph. They do not write storage, so delaying the FA
        // nodes until the join remains safe. Any other computation in this
        // interval could reuse an FA input or output allocation under the
        // original schedule, so fail closed instead of reordering it.
        for (int index = region.region_start; index <= region.join_node_index; ++index) {
            const ggml_tensor * node = ggml_graph_node(graph, index);
            if (branch_members.count(node) == 0 && concat_members.count(node) == 0 && !detail::is_view_or_noop(node)) {
                return detail::reject(
                    layer, "foreign compute node lies inside the packed region: index=" + std::to_string(index) +
                               " name=" + node->name + " op=" + ggml_op_name(node->op));
            }
        }

        for (int index = 0; index < node_count; ++index) {
            const ggml_tensor * consumer = ggml_graph_node(graph, index);
            for (int src = 0; src < GGML_MAX_SRC; ++src) {
                const ggml_tensor * producer = consumer->src[src];
                const auto          expected = expected_consumers.find(producer);
                if (expected != expected_consumers.end() && expected->second != consumer) {
                    return detail::reject(layer, "lane/concat node has an unexpected consumer");
                }
                if (concat_members.count(producer) != 0 && producer != group.join &&
                    expected == expected_consumers.end()) {
                    return detail::reject(layer, "concat node has an unexpected consumer");
                }
            }
        }

        for (const auto & lane : region.lanes) {
            for (size_t branch_index = 0; branch_index < lane.branch.size(); ++branch_index) {
                const ggml_tensor * node                  = lane.branch[branch_index];
                const ggml_tensor * allowed_branch_source = branch_index == 0 ? nullptr : lane.branch[branch_index - 1];
                for (int src = 0; src < GGML_MAX_SRC; ++src) {
                    if (branch_members.count(node->src[src]) != 0 && node->src[src] != allowed_branch_source) {
                        return detail::reject(layer, "lane has a cross-lane dependency");
                    }
                    if (concat_members.count(node->src[src]) != 0) {
                        return detail::reject(layer, "lane depends on its concat region");
                    }
                }
            }
        }

        // Preserve the allocator-visible relative order of independent concat
        // nodes rather than replaying the DFS traversal order.
        std::sort(region.concat_postorder.begin(), region.concat_postorder.end(),
                  [&](const ggml_tensor * lhs, const ggml_tensor * rhs) {
                      return node_indices.at(lhs) < node_indices.at(rhs);
                  });
        if (region.concat_postorder.back() != group.join) {
            return detail::reject(layer, "join is not the final concat node in graph order");
        }

        result.regions.push_back(std::move(region));
    }

    std::sort(result.regions.begin(), result.regions.end(),
              [](const auto & lhs, const auto & rhs) { return lhs.join_node_index < rhs.join_node_index; });
    return result;
}

}  // namespace ggml_vk_d2f
