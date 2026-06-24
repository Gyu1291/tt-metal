// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {

ttnn::Tensor transpose(
    const ttnn::Tensor& input_tensor,
    int64_t dim1,
    int64_t dim2,
    const std::optional<MemoryConfig>& memory_config_arg,
    float pad_value = 0.0f,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);

ttnn::Tensor transpose(
    const ttnn::Tensor& input_tensor,
    int64_t dim1,
    int64_t dim2,
    float pad_value = 0.0f,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);

}  // namespace ttnn
