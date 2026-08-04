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

#ifndef HIAI_API_INFRA_BASE_STATIC_STORAGE_H
#define HIAI_API_INFRA_BASE_STATIC_STORAGE_H

namespace hiai {
template <typename T>
struct StaticStorage {
    void* Alloc() noexcept
    {
        return static_cast<void*>(&u);
    }

    void Free() noexcept
    {
        Pointer()->~T();
    }

    const T* operator->() const noexcept
    {
        return Pointer();
    }

    T* operator->() noexcept
    {
        return Pointer();
    }

    const T& operator*() const noexcept
    {
        return Ref();
    }

    T& operator*() noexcept
    {
        return Ref();
    }

    const T* Pointer() const noexcept
    {
        return static_cast<const T*>(static_cast<const void*>(&u));
    }

    T* Pointer() noexcept
    {
        return static_cast<T*>(static_cast<void*>(&u));
    }

    const T& Ref() const noexcept
    {
        return *Pointer();
    }

    T& Ref() noexcept
    {
        return *Pointer();
    }

private:
    typename std::aligned_storage<sizeof(T), alignof(T)>::type u;
};
}

#endif // HIAI_API_INFRA_BASE_STATIC_STORAGE_H