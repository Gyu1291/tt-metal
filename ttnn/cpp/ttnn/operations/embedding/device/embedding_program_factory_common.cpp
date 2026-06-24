// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_program_factory_common.hpp"

namespace ttnn::prim {

namespace {

CoreRangeSet make_core_range_set(const std::vector<CoreCoord>& cores) {
    if (cores.empty()) {
        return CoreRangeSet();
    }
    return CoreRangeSet{ttsl::Span<const CoreCoord>(cores)};
}

}  // namespace

CoreSplitResult split_work_to_cores_aligned(
    const CoreCoord grid_size, const uint32_t units_to_divide, const uint32_t alignment) {
    ZoneScoped;

    uint32_t num_cores_x = grid_size.x, num_cores_y = grid_size.y;
    uint32_t total_cores = num_cores_x * num_cores_y;

    // Initialize units_per_core and required_cores
    uint32_t units_per_core = alignment;
    uint32_t required_cores = (units_to_divide + units_per_core - 1) / units_per_core;

    // find units per core and required cores
    if (required_cores > total_cores) {
        units_per_core = ((units_to_divide + total_cores - 1) / total_cores + alignment - 1) / alignment * alignment;
        required_cores = (units_to_divide + units_per_core - 1) / units_per_core;
    }

    // Core set for all active cores
    CoreRangeSet all_cores = tt::tt_metal::num_cores_to_corerangeset(required_cores, grid_size, false);

    // Calculate remaining units for the last core
    uint32_t evenly_distributed_units = (required_cores - 1) * units_per_core;
    uint32_t remaining_units = units_to_divide - evenly_distributed_units;

    // Create core groups
    CoreRangeSet core_group_1 = all_cores;
    CoreRangeSet core_group_2;

    // Handle the last core if remaining units are less than units_per_core
    if (remaining_units > 0 && remaining_units < units_per_core) {
        uint32_t last_core_x = (required_cores - 1) % num_cores_x;
        uint32_t last_core_y = (required_cores - 1) / num_cores_x;

        core_group_2 =
            CoreRangeSet(CoreRange(CoreCoord(last_core_x, last_core_y), CoreCoord(last_core_x, last_core_y)));
        core_group_1 = tt::tt_metal::num_cores_to_corerangeset(required_cores - 1, grid_size, false);
    }

    // Adjust the units per core for each group
    uint32_t units_per_core_group_1 = units_per_core;
    uint32_t units_per_core_group_2 = remaining_units < units_per_core ? remaining_units : 0;

    return CoreSplitResult{
        required_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2};
}

CoreSplitResult split_embedding_work_to_cores(
    tt::tt_metal::IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const CoreCoord grid_size,
    const uint32_t units_to_divide,
    const bool row_wise) {
    ZoneScoped;

    if (!sub_device_id.has_value()) {
        auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid_size, units_to_divide, row_wise);
        return CoreSplitResult{
            num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2};
    }

    if (units_to_divide == 0) {
        return CoreSplitResult{};
    }

    const auto sub_device_cores =
        device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
    const uint32_t max_num_cores = sub_device_cores.num_cores();
    TT_FATAL(max_num_cores > 0, "Sub-device core grid must contain at least one core");

    const uint32_t required_cores = std::min(units_to_divide, max_num_cores);
    const auto selected_cores = tt::tt_metal::corerange_to_cores(sub_device_cores, required_cores, row_wise);
    const CoreRangeSet all_cores = make_core_range_set(selected_cores);

    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = units_to_divide / required_cores;
    uint32_t units_per_core_group_2 = 0;
    const uint32_t num_cores_with_more_work = units_to_divide % required_cores;

    if (num_cores_with_more_work == 0) {
        core_group_1 = all_cores;
    } else {
        units_per_core_group_2 = units_per_core_group_1;
        units_per_core_group_1++;

        const std::vector<CoreCoord> group_1_cores(
            selected_cores.begin(), selected_cores.begin() + num_cores_with_more_work);
        const std::vector<CoreCoord> group_2_cores(
            selected_cores.begin() + num_cores_with_more_work, selected_cores.end());
        core_group_1 = make_core_range_set(group_1_cores);
        core_group_2 = make_core_range_set(group_2_cores);
    }

    return CoreSplitResult{
        required_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2};
}

CoreSplitResult split_embedding_work_to_cores_aligned(
    tt::tt_metal::IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const CoreCoord grid_size,
    const uint32_t units_to_divide,
    const uint32_t alignment) {
    ZoneScoped;

    if (!sub_device_id.has_value()) {
        return split_work_to_cores_aligned(grid_size, units_to_divide, alignment);
    }

    if (units_to_divide == 0) {
        return CoreSplitResult{};
    }

    const auto sub_device_cores =
        device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
    const uint32_t total_cores = sub_device_cores.num_cores();
    TT_FATAL(total_cores > 0, "Sub-device core grid must contain at least one core");

    uint32_t units_per_core = alignment;
    uint32_t required_cores = (units_to_divide + units_per_core - 1) / units_per_core;
    if (required_cores > total_cores) {
        units_per_core = ((units_to_divide + total_cores - 1) / total_cores + alignment - 1) / alignment * alignment;
        required_cores = (units_to_divide + units_per_core - 1) / units_per_core;
    }

    const auto selected_cores = tt::tt_metal::corerange_to_cores(sub_device_cores, required_cores, false);
    const CoreRangeSet all_cores = make_core_range_set(selected_cores);

    const uint32_t evenly_distributed_units = (required_cores - 1) * units_per_core;
    const uint32_t remaining_units = units_to_divide - evenly_distributed_units;

    CoreRangeSet core_group_1 = all_cores;
    CoreRangeSet core_group_2;
    if (remaining_units > 0 && remaining_units < units_per_core) {
        const std::vector<CoreCoord> group_1_cores(selected_cores.begin(), selected_cores.end() - 1);
        const std::vector<CoreCoord> group_2_cores(selected_cores.end() - 1, selected_cores.end());
        core_group_1 = make_core_range_set(group_1_cores);
        core_group_2 = make_core_range_set(group_2_cores);
    }

    const uint32_t units_per_core_group_1 = units_per_core;
    const uint32_t units_per_core_group_2 = remaining_units < units_per_core ? remaining_units : 0;

    return CoreSplitResult{
        required_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2};
}
}  // namespace ttnn::prim
