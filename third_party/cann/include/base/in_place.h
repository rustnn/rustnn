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

#ifndef HIAI_API_INFRA_BASE_IN_PLACE_H
#define HIAI_API_INFRA_BASE_IN_PLACE_H

#include "base/inline_variable.h"

namespace hiai {
struct InPlace {
    struct InPlaceTag {};

    constexpr explicit InPlace(InPlaceTag) {}
};

HIAI_INLINE_VARIABLE(InPlace, IN_PLACE, InPlace{InPlace::InPlaceTag{}});
} // namespace hiai

#endif // HIAI_API_INFRA_BASE_IN_PLACE_H