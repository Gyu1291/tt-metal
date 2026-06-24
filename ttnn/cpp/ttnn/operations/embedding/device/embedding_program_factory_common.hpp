// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <enchantum/enchantum.hpp>
#include "ttnn/operations/core/core.hpp"
#include "embedding_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <optional>
#include <vector>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {
struct CoreSplitResult {
    uint32_t required_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
};

CoreSplitResult split_work_to_cores_aligned(CoreCoord grid_size, uint32_t units_to_divide, uint32_t alignment);
CoreSplitResult split_embedding_work_to_cores(
    tt::tt_metal::IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    CoreCoord grid_size,
    uint32_t units_to_divide,
    bool row_wise = false);
CoreSplitResult split_embedding_work_to_cores_aligned(
    tt::tt_metal::IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    CoreCoord grid_size,
    uint32_t units_to_divide,
    uint32_t alignment);
}  // namespace ttnn::prim
