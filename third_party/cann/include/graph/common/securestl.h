/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __COMMON_SECURESTL_H__
#define __COMMON_SECURESTL_H__
#include <memory>
#include "infra/alias/aicp.h"

namespace hiai {
template <typename T, typename... Ts>
std::shared_ptr<T> make_shared_nothrow(Ts&&... params)
{
    return std::shared_ptr<T>(new (std::nothrow) T(std::forward<Ts>(params)...));
}

template <typename T, typename... Ts>
std::unique_ptr<T> make_unique_nothrow(Ts&&... params)
{
    return std::unique_ptr<T>(new (std::nothrow) T(std::forward<Ts>(params)...));
}
} // namespace hiai
#endif
