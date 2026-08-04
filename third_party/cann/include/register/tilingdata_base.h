/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef __INC_REGISTER_ASCENDC_TILINGDATA_BASE_HEADER__
#define __INC_REGISTER_ASCENDC_TILINGDATA_BASE_HEADER__

#include <vector>
#include <map>
#include <memory>
#include <cstring>
#include <securec.h>
#include "graph/ascend_string.h"

namespace optiling {
struct CharPtrCmp {
  bool operator()(const char *strLeft, const char *strRight) const
  {
    return strcmp(strLeft, strRight) < 0;
  }
};

class StructSizeInfoBase {
public:
  static StructSizeInfoBase &GetInstance();

  void SetStructSize(const char *structType, const size_t structSize)
  {
    if (structSizeInfo.find(structType) != structSizeInfo.end()) {
      return;
    }
    structSizeInfo[structType] = structSize;
  }
  size_t GetStructSize(const char *structType)
  {
    return structSizeInfo.at(structType);
  }
private:
  StructSizeInfoBase() { };
  ~StructSizeInfoBase() { };
  StructSizeInfoBase(const StructSizeInfoBase &);
  StructSizeInfoBase &operator=(const StructSizeInfoBase &);
  std::map<const char *, size_t, CharPtrCmp> structSizeInfo;
};

class FieldInfo {
public:
  FieldInfo(const char *dtype, const char *name)
      : dtype_(dtype), name_(name), classType_("0") {}
  FieldInfo(const char *dtype, const char *name, size_t arrSize)
      : dtype_(dtype), name_(name), arrSize_(arrSize), classType_("1") {}
  FieldInfo(const char *dtype, const char *name, const char *structType,
            size_t structSize)
      : dtype_(dtype), name_(name), structType_(structType), structSize_(structSize), classType_("2") {}

public:
  const char *dtype_;
  const char *name_;
  size_t arrSize_;
  const char *structType_;
  size_t structSize_;
  const char *classType_;
};

class TilingDef {
public:
  ~TilingDef()
  {
    if (!inited_data_ptr && data_ptr_ != nullptr) {
      delete[] data_ptr_;
    }
    data_ptr_ = nullptr;
    class_name_ = nullptr;
  }
  void SaveToBuffer(void *pdata, size_t capacity);
  std::vector<FieldInfo> GetFieldInfo() const;
  const char *GetTilingClassName() const;
  size_t GetDataSize() const;
  void SetDataPtr(void *dataPtr);
  void CheckAlignAndGenPlaceHolder(const char *name, size_t typeSize);
protected:
  void InitData();
  void GeLogError(const std::string& str) const;
  // dtype, name
  std::vector<FieldInfo> field_info_;
  uint8_t *data_ptr_ = nullptr;
  size_t data_size_ = 0;
  const char *class_name_;
  std::vector<std::pair<void *, size_t>> saveBufferPtr;
  size_t struct_size_ = 0;
  bool inited_data_ptr = false;
  uint32_t feature_bit_flag = 0;
  uint8_t reserved_buf[128] = {0};
};

using TilingDataConstructor = std::shared_ptr<TilingDef> (*)();

class CTilingDataClassFactory {
public:
  static CTilingDataClassFactory &GetInstance();
  void RegisterTilingData(const char *op_type, const TilingDataConstructor constructor);
  std::shared_ptr<TilingDef> CreateTilingDataInstance(const char *op_type);

private:
  CTilingDataClassFactory() { };
  ~CTilingDataClassFactory() { };
  CTilingDataClassFactory(const CTilingDataClassFactory &);
  CTilingDataClassFactory &operator=(const CTilingDataClassFactory &);
  std::map<const char *, TilingDataConstructor, CharPtrCmp> instance_;
};

class TilingDataStructBase {
public:
  static TilingDataStructBase &GetInstance()
  {
    static TilingDataStructBase instance;
    return instance;
  }
  uint32_t __attribute__((weak)) RecordTilingStruct(const char* name, const char* file, uint32_t line);
private:
  TilingDataStructBase() { };
  ~TilingDataStructBase() { };
  TilingDataStructBase(const TilingDataStructBase &);
  TilingDataStructBase &operator=(const TilingDataStructBase &);
  std::map<const char *, std::pair<const char *, uint32_t>, CharPtrCmp> records;
};
}  // end of namespace optiling

/*
example:
// supported data_type: int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t
BEGIN_TILING_DATA_DEF(MaxPoolTilingData)
    // format: TILING_DATA_FIELD_DEF(data_type, field_name);
    TILING_DATA_FIELD_DEF(int32_t, dim_0);
    TILING_DATA_FIELD_DEF(uint8_t, var_1);
    TILING_DATA_FIELD_DEF(int64_t, factor_1);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MaxPool, MaxPoolTilingData)
*/
#define BEGIN_TILING_DATA_DEF(class_name)                                                                              \
namespace {                                                                                                            \
  static uint32_t class_name##tiling = [](void) {                                                                      \
    if (&TilingDataStructBase::RecordTilingStruct != nullptr) {                                                        \
      return TilingDataStructBase::GetInstance().RecordTilingStruct(#class_name, __FILE__, __LINE__);                  \
    } else {                                                                                                           \
      return static_cast<uint32_t>(0);                                                                                 \
    }                                                                                                                  \
  }();                                                                                                                 \
}                                                                                                                      \
  class class_name : public TilingDef {                                                                                \
   public:                                                                                                             \
    size_t FieldHandler(const char *dtype, const char *name, size_t typeSize, const char* namePh) {                    \
      CheckAlignAndGenPlaceHolder(namePh, typeSize);                                                                   \
      field_info_.emplace_back(FieldInfo(dtype, name));                                                                \
      size_t ret_val = data_size_;                                                                                     \
      data_size_ += typeSize;                                                                                          \
      return ret_val;                                                                                                  \
    }                                                                                                                  \
    size_t FieldHandler(const char *dtype, const char *name, size_t typeSize,                                          \
                 size_t arrSize, const char* namePh) {                                                                 \
      CheckAlignAndGenPlaceHolder(namePh, typeSize);                                                                   \
      field_info_.emplace_back(FieldInfo(dtype, name, arrSize));                                                       \
      size_t ret_val = data_size_;                                                                                     \
      data_size_ += typeSize * arrSize;                                                                                \
      return ret_val;                                                                                                  \
    }                                                                                                                  \
    size_t FieldHandler(const char *dtype, const char *name,                                                           \
                 const char *structType, size_t structSize, void *ptr, const char* namePh) {                           \
      CheckAlignAndGenPlaceHolder(namePh, 8);                                                                          \
      field_info_.emplace_back(FieldInfo(dtype, name, structType, structSize));                                        \
      size_t ret_val = data_size_;                                                                                     \
      data_size_ += structSize;                                                                                        \
      saveBufferPtr.emplace_back(std::make_pair(ptr, ret_val));                                                        \
      struct_size_ += structSize;                                                                                      \
      return ret_val;                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
   public:                                                                                                             \
    class_name() {                                                                                                     \
      class_name_ = #class_name;                                                                                       \
      CheckAlignAndGenPlaceHolder(#class_name"PH", 8);                                                                 \
      StructSizeInfoBase::GetInstance().SetStructSize(#class_name, data_size_);                                        \
      InitData();                                                                                                      \
    }                                                                                                                  \
    explicit class_name(void *ptr) {                                                                                   \
      class_name_ = #class_name;                                                                                       \
      CheckAlignAndGenPlaceHolder(#class_name"PH", 8);                                                                 \
      StructSizeInfoBase::GetInstance().SetStructSize(#class_name, data_size_);                                        \
      if (ptr == nullptr) {                                                                                            \
        return;                                                                                                        \
      }                                                                                                                \
      SetDataPtr(ptr);                                                                                                 \
    }

#define TILING_DATA_FIELD_DEF(data_type, field_name)                                                                   \
 public:                                                                                                               \
  void set_##field_name(data_type field_name) {                                                                        \
    field_name##_ = field_name;                                                                                        \
    *((data_type *) (data_ptr_ + field_name##_offset_)) = field_name;                                                  \
  }                                                                                                                    \
  data_type get_##field_name() { return field_name##_; }                                                               \
                                                                                                                       \
 private:                                                                                                              \
  data_type field_name##_ = 0;                                                                                         \
  size_t field_name##_offset_ = FieldHandler(#data_type, #field_name, sizeof(data_type), #field_name"PH");             \
  __attribute__((unused)) uint8_t field_name##_reserve_buf_[16] = {0};

#define TILING_DATA_FIELD_DEF_ARR(arr_type, arr_size, field_name)                                                      \
 public:                                                                                                               \
  void set_##field_name(arr_type *field_name) {                                                                        \
    field_name##_ = field_name;                                                                                        \
    auto offset = field_name##_offset_;                                                                                \
    if (data_ptr_ + offset == (uint8_t *)field_name) {                                                                 \
      return;                                                                                                          \
    }                                                                                                                  \
    const auto err_t = memcpy_s(data_ptr_ + offset, data_size_ - offset, field_name, (arr_size) * sizeof(arr_type));   \
    if (err_t != EOK) {                                                                                                \
        GeLogError("tilingdata_base.h TILING_DATA_FIELD_DEF_ARR memcpy is failed !");                                  \
    }                                                                                                                  \
  }                                                                                                                    \
  arr_type *get_##field_name() {                                                                                       \
    return (arr_type *)(data_ptr_ + field_name##_offset_);                                                             \
  }                                                                                                                    \
                                                                                                                       \
 private:                                                                                                              \
  arr_type *field_name##_ = nullptr;                                                                                   \
  size_t field_name##_offset_ = FieldHandler(#arr_type, #field_name, sizeof(arr_type), arr_size, #field_name"PH");     \
  __attribute__((unused)) uint8_t field_name##_reserve_buf_[16] = {0};

#define TILING_DATA_FIELD_DEF_STRUCT(struct_type, field_name)                                                          \
 public:                                                                                                               \
  struct_type field_name{nullptr};                                                                                     \
                                                                                                                       \
 private:                                                                                                              \
  size_t field_name##_offset_ =                                                                                        \
      FieldHandler("struct", #field_name, #struct_type,                                                                \
                   StructSizeInfoBase::GetInstance().GetStructSize(#struct_type),                                      \
                   (void *) &field_name, #field_name"PH");                                                             \
  __attribute__((unused)) uint8_t field_name##_reserve_buf_[16] = {0};

#define END_TILING_DATA_DEF                                                                                            \
  }                                                                                                                    \
  ;

#define REGISTER_TILING_DATA_CLASS(op_type, class_name)                                                                \
namespace {                                                                                                            \
  class op_type##class_name##Helper {                                                                                  \
   public:                                                                                                             \
    op_type##class_name##Helper() {                                                                                    \
      CTilingDataClassFactory::GetInstance().RegisterTilingData(#op_type,                                              \
                   op_type##class_name##Helper::CreateTilingDataInstance);                                             \
    }                                                                                                                  \
    static std::shared_ptr<TilingDef> CreateTilingDataInstance() { return std::make_shared<class_name>(); }            \
  };                                                                                                                   \
  static class_name g_##op_type##class_name##init;                                                                     \
  static op_type##class_name##Helper g_tilingdata_##op_type##class_name##helper;                                       \
}
#endif  // __INC_REGISTER_ASCENDC_TILINGDATA_BASE_HEADER__
