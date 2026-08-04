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
#ifndef FRAMEWORK_TENSOR_TENSOR_API_EXPORT_H
#define FRAMEWORK_TENSOR_TENSOR_API_EXPORT_H

#ifdef HIAI_TENSOR_API_VISIABLE
#ifdef _MSC_VER
#define HIAI_TENSOR_API_EXPORT __declspec(dllexport)
#else
#define HIAI_TENSOR_API_EXPORT __attribute__((__visibility__("default")))
#endif
#else
#define HIAI_TENSOR_API_EXPORT
#endif

#endif // FRAMEWORK_TENSOR_TENSOR_API_EXPORT_H
