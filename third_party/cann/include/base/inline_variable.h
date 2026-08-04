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

#ifndef HIAI_API_INFRA_BASE_INLINE_VARIABLE_H
#define HIAI_API_INFRA_BASE_INLINE_VARIABLE_H

#include "base/identity.h"

#define HIAI_INLINE_VARIABLE(type, name, val)                                                                          \
    template <typename = void>                                                                                         \
    struct InlineVariable##name {                                                                                      \
        constexpr static IdentityType<type> value = val;                                                               \
    };                                                                                                                 \
                                                                                                                       \
    template <typename T>                                                                                              \
    constexpr IdentityType<type> InlineVariable##name<T>::value;                                                       \
                                                                                                                       \
    constexpr static const IdentityType<type>& (name) = InlineVariable##name<>::value

#endif // HIAI_API_INFRA_BASE_INLINE_VARIABLE_H