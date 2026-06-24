// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::prim {

inline std::tuple<uint32_t, tt::tt_metal::CoreRangeSet, tt::tt_metal::CoreRangeSet, tt::tt_metal::CoreRangeSet, uint32_t, uint32_t>
split_copy_work_to_cores(
    tt::tt_metal::IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const tt::tt_metal::CoreCoord& compute_with_storage_grid_size,
    uint32_t units_to_divide,
    bool row_wise = false) {
    using namespace tt::tt_metal;

    if (!sub_device_id.has_value()) {
        return split_work_to_cores(compute_with_storage_grid_size, units_to_divide, row_wise);
    }

    if (units_to_divide == 0) {
        return std::make_tuple(0, CoreRangeSet(), CoreRangeSet(), CoreRangeSet(), 0, 0);
    }

    const auto sub_device_cores = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id.value());
    const uint32_t max_num_cores = sub_device_cores.num_cores();
    TT_FATAL(max_num_cores > 0, "Sub-device core grid must contain at least one core");

    const uint32_t target_num_cores = std::min(units_to_divide, max_num_cores);
    const auto selected_cores_vec = corerange_to_cores(sub_device_cores, target_num_cores, row_wise);
    const CoreRangeSet all_cores{ttsl::Span<const CoreCoord>(selected_cores_vec)};

    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = units_to_divide / target_num_cores;
    uint32_t units_per_core_group_2 = 0;
    const uint32_t num_cores_with_more_work = units_to_divide % target_num_cores;

    if (num_cores_with_more_work == 0) {
        core_group_1 = all_cores;
    } else {
        units_per_core_group_2 = units_per_core_group_1;
        units_per_core_group_1++;

        std::vector<CoreCoord> group_1_cores(
            selected_cores_vec.begin(), selected_cores_vec.begin() + num_cores_with_more_work);
        std::vector<CoreCoord> group_2_cores(
            selected_cores_vec.begin() + num_cores_with_more_work, selected_cores_vec.end());
        core_group_1 = CoreRangeSet{ttsl::Span<const CoreCoord>(group_1_cores)};
        core_group_2 = CoreRangeSet{ttsl::Span<const CoreCoord>(group_2_cores)};
    }

    return std::make_tuple(
        target_num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2);
}

}  // namespace ttnn::prim
