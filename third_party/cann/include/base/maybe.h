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

#ifndef HIAI_API_INFRA_BASE_MAYBE_H
#define HIAI_API_INFRA_BASE_MAYBE_H

#include <utility>

#include "base/inline_variable.h"
#include "base/static_storage.h"
#include "base/in_place.h"

namespace hiai {
struct NullMaybe {
    struct NullMaybeTag {};

    constexpr explicit NullMaybe(NullMaybeTag) {}
};

HIAI_INLINE_VARIABLE(NullMaybe, NULL_MAYBE, NullMaybe{NullMaybe::NullMaybeTag{}});

template <typename T>
struct Maybe {
    constexpr Maybe() noexcept = default;

    constexpr explicit Maybe(NullMaybe) noexcept {}

    Maybe(const Maybe<T>& other) noexcept
    {
        if (other) {
            Construct(other.Ref());
        }
    }

    Maybe(Maybe<T>&& other) noexcept
    {
        if (other) {
            Construct(std::move(other.Ref()));
        }
    }

    explicit Maybe(const T& other) noexcept : exist_(true)
    {
        Construct(other);
    }

    explicit Maybe(T&& other) noexcept : exist_(true)
    {
        Construct(std::move(other));
    }

    template <typename... Args>
    explicit Maybe(InPlace, Args&&... args) noexcept
    {
        Construct(std::forward<Args>(args)...);
    }

    template <typename U, typename... Args>
    explicit Maybe(InPlace, std::initializer_list<U> il, Args&&... args) noexcept
    {
        Construct(il, std::forward<Args>(args)...);
    }

    ~Maybe() noexcept
    {
        Clear();
    }

    Maybe& operator=(const Maybe<T>& other) noexcept
    {
        if (this == &other) {
            return *this;
        }
        if (other) {
            operator=(other.Ref());
        } else {
            Clear();
        }
        return *this;
    }

    Maybe& operator=(Maybe<T>&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }
        if (other) {
            operator=(std::move(other.Ref()));
        } else {
            Clear();
        }
        return *this;
    }

    Maybe& operator=(const T& other) noexcept
    {
        if (exist_) {
            Ref() = other;
        } else {
            Construct(other);
        }
        return *this;
    }

    Maybe& operator=(T&& other) noexcept
    {
        if (exist_) {
            Ref() = std::move(other);
        } else {
            Construct(std::move(other));
        }
        return *this;
    }

    template <typename... Args>
    void Emplace(Args&&... args) noexcept
    {
        Clear();
        Construct(std::forward<Args>(args)...);
    }

    template <class U, class... Args>
    void Emplace(std::initializer_list<U> il, Args&&... args) noexcept
    {
        Clear();
        Construct(il, std::forward<Args>(args)...);
    }

    void Swap(Maybe<T>& other) noexcept
    {
        if (*this) {
            if (other) {
                std::swap(Ref(), other.Ref());
            } else {
                other.Construct(std::move(Ref()));
                Destruct();
            }
        } else {
            if (other) {
                Construct(std::move(other.Ref()));
                other.Destruct();
            }
        }
    }

    const T& operator*() const noexcept
    {
        return *Pointer();
    }

    T& operator*() noexcept
    {
        return *Pointer();
    }

    const T* operator->() const noexcept
    {
        return Pointer();
    }

    T* operator->() noexcept
    {
        return Pointer();
    }

    const T& Value() const& noexcept
    {
        return Ref();
    }

    T& Value() & noexcept
    {
        return Ref();
    }

    T&& Value() && noexcept
    {
        return std::move(Ref());
    }

    const T&& Value() const&& noexcept
    {
        return std::move(Ref());
    }

    operator bool() const noexcept
    {
        return exist_;
    }

    bool Exist() const noexcept
    {
        return exist_;
    }

private:
    const T* Pointer() const noexcept
    {
        return storage_.Pointer();
    }

    T* Pointer() noexcept
    {
        return storage_.Pointer();
    }

    T& Ref() noexcept
    {
        return storage_.Ref();
    }

    const T& Ref() const noexcept
    {
        return storage_.Ref();
    }

    template <typename... Args>
    void Construct(Args&&... args) noexcept
    {
        exist_ = true;
        new (storage_.Alloc()) T(std::forward<Args>(args)...);
    }

    void Destruct() noexcept
    {
        storage_.Free();
        exist_ = false;
    }

    void Clear() noexcept
    {
        if (exist_) {
            Destruct();
        }
    }

private:
    StaticStorage<T> storage_;
    bool exist_{false};
};
} // namespace hiai

#endif // HIAI_API_INFRA_BASE_MAYBE_H