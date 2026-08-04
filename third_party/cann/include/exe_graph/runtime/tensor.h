/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_INC_EXE_GRAPH_TENSOR_H_
#define METADEF_CXX_INC_EXE_GRAPH_TENSOR_H_

#include <cstring>

#include "graph/ge_error_codes.h"
#include "storage_shape.h"
#include "storage_format.h"
#include "tensor_data.h"

namespace gert {
using TensorAddress = void *;            ///< Tensor地址
using ConstTensorAddress = void *const;  ///< Tensor地址

class Tensor {
 public:
  // memse函数misra告警屏蔽
  Tensor() {
    (void) memset(reserved_, 0, sizeof(reserved_));
    (void) memset(reserved_field_, 0, sizeof(reserved_field_));
  }
  Tensor(const StorageShape &storage_shape, const StorageFormat &storage_format, const TensorPlacement placement,
         const ge::DataType data_type, TensorAddress addr)
      : Tensor(storage_shape, storage_format, placement, data_type, addr, nullptr) {}
  Tensor(const StorageShape &storage_shape, const StorageFormat &storage_format, ge::DataType data_type)
      : storage_shape_(storage_shape), storage_format_(storage_format), data_type_(data_type) {
    (void) memset(reserved_, 0, sizeof(reserved_));
    (void) memset(reserved_field_, 0, sizeof(reserved_field_));
  }
  Tensor(const StorageShape &storage_shape, const StorageFormat &storage_format, const TensorPlacement placement,
         const ge::DataType data_type, TensorAddress addr, TensorAddrManager manager)
      : storage_shape_(storage_shape), storage_format_(storage_format), data_type_(data_type),
        tensor_data_(addr, manager, static_cast<size_t>(ge::GetSizeInBytes(GetShapeSize(), data_type_)), placement) {
    (void) memset(reserved_, 0, sizeof(reserved_));
    (void) memset(reserved_field_, 0, sizeof(reserved_field_));
  }
  /**
   * 获取shape size，所谓shape size是指本shape中包含的element数量
   * @return shape size
   */
  int64_t GetShapeSize() const {
    return storage_shape_.GetStorageShape().GetShapeSize();
  }
  /**
   * 获取Tensor的数据地址
   * @tparam T 数据类型
   * @return 数据地址
   */
  template<class T>
  const T *GetData() const {
    return static_cast<const T *>(GetAddr());
  }
  /**
   * 获取Tensor的数据地址
   * @tparam T 数据类型
   * @return 数据地址
   */
  template<class T>
  auto GetData() -> T* {
    return static_cast<T *>(GetAddr());
  }
  /**
   * 设置Tensor数据
   * @param data 数据
   */
  void SetData(TensorData &&data) {
    tensor_data_ = std::move(data);
  }
  /**
   * 获取Tensor的数据地址
   * @return 数据地址
   */
  const void *GetAddr() const {
    if (tensor_data_.GetPlacement() == kFollowing) {
      return reinterpret_cast<const void *>(reinterpret_cast<const uint8_t *>(this) + sizeof(*this));
    } else {
      return tensor_data_.GetAddr();
    }
  }
  /**
   * 获取Tensor的数据地址
   * @return 数据地址
   */
  void *GetAddr() {
    if (tensor_data_.GetPlacement() == kFollowing) {
      return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(this) + sizeof(*this));
    } else {
      return tensor_data_.GetAddr();
    }
  }
  /**
   * 获取Tensor的内存大小
   * @return 内存大小
   */
  size_t GetSize() const {
    return tensor_data_.GetSize();
  }

  /**
   * 设置Tensor的内存大小
   * @param Tensor的内存大小
   */
  void SetSize(const size_t size) {
    tensor_data_.SetSize(size);
  }

  /**
   * 获取Tensor的data type
   * @return data type
   */
  ge::DataType GetDataType() const {
    return data_type_;
  }
  /**
   * 设置Tensor的data type
   * @param data_type data type
   */
  void SetDataType(const ge::DataType data_type) {
    data_type_ = data_type;
  }
  /**
   * 创建一个Tensor，tensor数据在Tensor对象后面连续排布
   * @param shape_size 元素个数
   * @param dt 数据类型
   * @param total_size 创建出的Tensor在内存中的长度
   * @return 创建出的Tensor指针
   */
  static std::unique_ptr<uint8_t[]> CreateFollowing(const int64_t shape_size, const ge::DataType dt,
                                                    size_t &total_size) {
    total_size = static_cast<size_t>(ge::GetSizeInBytes(shape_size, dt));
    return NewFollowingTensor(dt, total_size);
  }
  /**
   * 创建一个Tensor，tensor数据在Tensor对象后面连续排布
   * @param tensor_size tensor长度
   * @param dt 数据类型
   * @param total_size 创建出的Tensor在内存中的长度
   * @return 创建出的Tensor指针
   */
  static std::unique_ptr<uint8_t[]> CreateFollowing(const ge::DataType dt, const size_t tensor_size,
                                                    size_t &total_size) {
    total_size = tensor_size;
    if (ge::AddOverflow(total_size, sizeof(Tensor), total_size)) {
      return nullptr;
    }
    auto holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
    if (holder == nullptr) {
      return nullptr;
    }
    auto tensor = reinterpret_cast<Tensor *>(holder.get());
    new (holder.get()) Tensor({}, {}, dt);
    tensor->SetPlacement(kFollowing);
    tensor->tensor_data_ = TensorData(nullptr, nullptr, total_size - sizeof(Tensor), kFollowing);
    return holder;
  }
  /**
   * 获取运行shape
   * @return 只读的运行时shape引用
   */
  const Shape &GetStorageShape() const {
    return storage_shape_.GetStorageShape();
  }
  /**
   * 获取运行shape
   * @return 运行时shape的引用
   */
  Shape &MutableStorageShape() {
    return storage_shape_.MutableStorageShape();
  }
  /**
   * 获取原始shape
   * @return 只读的原始shape引用
   */
  const Shape &GetOriginShape() const {
    return storage_shape_.GetOriginShape();
  }
  /**
   * 获取原始shape
   * @return 原始shape引用
   */
  Shape &MutableOriginShape() {
    return storage_shape_.MutableOriginShape();
  }
  /**
   * 获取shape，包含运行和原始shape
   * @return 只读的shape引用
   */
  const StorageShape &GetShape() const {
    return storage_shape_;
  }
  /**
   * 获取shape，包含运行和原始shape
   * @return shape引用
   */
  StorageShape &GetShape() {
    return storage_shape_;
  }
  /**
   * 获取运行时format
   * @return 运行时format
   */
  ge::Format GetStorageFormat() const {
    return storage_format_.GetStorageFormat();
  }
  /**
   * 设置运行时format
   * @param storage_format 运行时format
   */
  void SetStorageFormat(const ge::Format storage_format) {
    storage_format_.SetStorageFormat(storage_format);
  }
  /**
   * 获取原始format
   * @return 原始format
   */
  ge::Format GetOriginFormat() const {
    return storage_format_.GetOriginFormat();
  }
  /**
   * 设置原始format
   * @param origin_format 原始format
   */
  void SetOriginFormat(const ge::Format origin_format) {
    storage_format_.SetOriginFormat(origin_format);
  }
  /**
   * 获取format，format包含运行时format和原始format
   * @return 只读的format引用
   */
  const StorageFormat &GetFormat() const {
    return storage_format_;
  }
  /**
   * 获取format，format包含运行时format和原始format
   * @return format引用
   */
  StorageFormat &MutableFormat() {
    return storage_format_;
  }
  /**
   * 获取补维规则
   * @return 补维规则
   */
  ExpandDimsType GetExpandDimsType() const {
    return storage_format_.GetExpandDimsType();
  }
  /**
   * 设置补维规则
   * @param expand_dims_type 补维规则
   */
  void SetExpandDimsType(const ExpandDimsType &expand_dims_type) {
    storage_format_.SetExpandDimsType(expand_dims_type);
  }
  /**
   * 获取tensor的placement
   * @return tensor的placement
   */
  TensorPlacement GetPlacement() const {
    return tensor_data_.GetPlacement();
  }
  /**
   * 设置tensor的placement
   * @param tensor的placement
   */
  void SetPlacement(const TensorPlacement placement) {
    tensor_data_.SetPlacement(placement);
  }
  /**
   * 获取tensor data
   * @return 只读的tensor data引用
   */
  const TensorData &GetTensorData() const {
    return tensor_data_;
  }
  /**
   * 获取tensor data
   * @return 可写的tensor data引用
   */
  TensorData &MutableTensorData() {
    return tensor_data_;
  }
 private:
  static std::unique_ptr<uint8_t[]> NewFollowingTensor(const ge::DataType dt, size_t &total_size) {
    if (ge::AddOverflow(total_size, sizeof(Tensor), total_size)) {
      return nullptr;
    }
    auto holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
    if (holder == nullptr) {
      return nullptr;
    }
    auto tensor = reinterpret_cast<Tensor *>(holder.get());
    new (holder.get()) Tensor({}, {},  dt);
    tensor->SetPlacement(kFollowing);
    tensor->tensor_data_ = TensorData(nullptr, nullptr, total_size - sizeof(Tensor), kFollowing);
    return holder;
  }
 private:
  StorageShape storage_shape_;
  StorageFormat storage_format_;
  uint8_t reserved_[4]; // Reserved field, 4-byte aligned
  ge::DataType data_type_;
  TensorData tensor_data_;
  uint8_t reserved_field_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<Tensor>::value, "The class Tensor must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_TENSOR_H_
