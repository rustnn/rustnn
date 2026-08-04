/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef OP_DEF_H
#define OP_DEF_H

#include <iostream>
#include <vector>
#include <memory>
#include "register/op_impl_registry.h"
#include "graph/operator_reg.h"

namespace optiling {
#define FUNC_CHECK_SUPPORTED "check_supported"
#define FUNC_OP_SELECT_FORMAT "op_select_format"
#define FUNC_GET_OP_SUPPORT_INFO "get_op_support_info"
#define FUNC_GET_SPECIFIC_INFO "get_op_specific_info"

using OP_CHECK_FUNC = ge::graphStatus (*)(const ge::Operator &op, ge::AscendString &result);

using PARAM_GENERALIZE_FUNC = ge::graphStatus (*)(const ge::Operator &op, const ge::AscendString &generalize_config,
                                      ge::AscendString &generalized_op_params);

class OpCheckFuncHelper {
public:
  OpCheckFuncHelper(const ge::AscendString &check_type, const ge::AscendString &op_type, OP_CHECK_FUNC func);

  OpCheckFuncHelper(const ge::AscendString &op_type, PARAM_GENERALIZE_FUNC func);
};
}

namespace ops {
class AclnnOpGenerator;
class Generator;
class OpProtoGenerator;
class GeneratorFactory;
class CfgGenerator;
class OpParamTrunk;
class OpDefUtils;

enum Option { IGNORE = 0, OPTIONAL = 1, REQUIRED = 2, DYNAMIC = 3, VIRTUAL = 4 };

enum class DependScope : uint32_t {
  ALL = 0,
  TILING = 1,
  INVALID_SCOPE
};

enum class FollowType : uint32_t {
  ALL = 0,
  DTYPE = 1,
  FORMAT = 2,
  SHAPE = 3,
  INVALID_TYPE
};

enum class AttrDataType {
  ATTR_DT_BOOL = 0,
  ATTR_DT_FLOAT = 1,
  ATTR_DT_INT = 2,
  ATTR_DT_STR = 3,
  ATTR_DT_LIST_BOOL = 4,
  ATTR_DT_LIST_FLOAT = 5,
  ATTR_DT_LIST_INT = 6,
  ATTR_DT_LIST_LIST_INT = 7,
  ATTR_DT_MAX
};

enum class InitValueType : uint32_t {
  INIT_VALUE_UINT64_T = 0,
  INIT_VALUE_DEFAULT = static_cast<uint32_t>(-1),
};

enum class CommentSection : uint32_t {
  CATEGORY = 0,
  BRIEF = 1,
  CONSTRAINTS = 2,
  RESTRICTIONS = 3,
  SEE = 4,
  THIRDPARTYFWKCOMPAT = 5,
  SECTION_MAX
};

enum class ScalarType : uint32_t {
  UINT64 = 0,
  INT64 = 1,
  UINT32 = 2,
  INT32 = 3,
  UINT16 = 4,
  INT16 = 5,
  UINT8 = 6,
  INT8 = 7,
  FLOAT32 = 8,
  FLOAT16 = 9,
  INVALID_DTYPE = static_cast<uint32_t>(-1),
};

union ScalarNum {
  uint64_t value_u64;
  int64_t value_i64;
  float value_f32;
  ScalarNum() : value_u64(0) {}
  explicit ScalarNum(uint64_t value) : value_u64(value) {}
  explicit ScalarNum(int64_t value) : value_i64(value) {}
  explicit ScalarNum(float value) : value_f32(value) {}
};

using InitValueNum = ScalarNum;

struct ScalarVar {
  ScalarType scalar_type;
  ScalarNum scalar_num;
  ScalarVar() : scalar_type(ScalarType::INVALID_DTYPE) {}
  ScalarVar(ScalarType type, uint64_t num) : scalar_type(type), scalar_num(num) {
    if (type == ScalarType::FLOAT32 || type == ScalarType::FLOAT16) {
      scalar_num = ScalarNum(static_cast<float>(num));
    }
  }
  ScalarVar(ScalarType type, int64_t num) : scalar_type(type), scalar_num(num) {
    if (type == ScalarType::FLOAT32 || type == ScalarType::FLOAT16) {
      scalar_num = ScalarNum(static_cast<float>(num));
    }
  }
  ScalarVar(ScalarType type, int num) : scalar_type(type), scalar_num(static_cast<int64_t>(num)) {
    if (type == ScalarType::FLOAT32 || type == ScalarType::FLOAT16) {
      scalar_num = ScalarNum(static_cast<float>(num));
    }
  }
  ScalarVar(ScalarType type, unsigned int num) : scalar_type(type), scalar_num(static_cast<uint64_t>(num)) {
    if (type == ScalarType::FLOAT32 || type == ScalarType::FLOAT16) {
      scalar_num = ScalarNum(static_cast<float>(num));
    }
  }
  ScalarVar(ScalarType type, float num) : scalar_type(type), scalar_num(num) {
    if (type != ScalarType::FLOAT32 && type != ScalarType::FLOAT16) {
      if (type == ScalarType::UINT64) {
        scalar_num = ScalarNum(static_cast<uint64_t>(num));
      }
      scalar_num = ScalarNum(static_cast<int64_t>(num));
    }
  }
  ScalarVar(ScalarType type, double num) : scalar_type(type), scalar_num(static_cast<float>(num)) {
    if (type != ScalarType::FLOAT32 && type != ScalarType::FLOAT16) {
      if (type == ScalarType::UINT64) {
        scalar_num = ScalarNum(static_cast<uint64_t>(num));
      }
      scalar_num = ScalarNum(static_cast<int64_t>(num));
    }
  }
  bool operator==(const ScalarVar& other) const {
    if (scalar_type == other.scalar_type && scalar_num.value_u64 == other.scalar_num.value_u64) {
      return true;
    }
    return false;
  }
};

enum class ItemFindStatus { ITEM_FIND = 0, ITEM_NOEXIST = 1 };

class OpParamDefImpl;
class OpParamDef {
public:
  explicit OpParamDef(const char *name);
  OpParamDef(const OpParamDef &def);
  ~OpParamDef();
  OpParamDef &operator=(const OpParamDef &def);
  OpParamDef &ParamType(Option param_type);
  OpParamDef &DataType(std::vector<ge::DataType> types);
  OpParamDef &DataTypeList(std::vector<ge::DataType> types);
  OpParamDef &Format(std::vector<ge::Format> formats);
  OpParamDef &FormatList(std::vector<ge::Format> formats);
  OpParamDef &UnknownShapeFormat(std::vector<ge::Format> formats);
  OpParamDef &ValueDepend(Option value_depend);
  OpParamDef &ValueDepend(Option value_depend, DependScope scope);
  OpParamDef &IgnoreContiguous(void);
  OpParamDef &AutoContiguous();
  OpParamDef &Scalar();
  OpParamDef &ScalarList();
  OpParamDef &To(const ge::DataType type);
  OpParamDef &To(const char *name);
  OpParamDef &Version(uint32_t version);
  OpParamDef &InitValue(uint64_t value);
  OpParamDef &InitValue(const ScalarVar &value);
  OpParamDef &InitValue(const std::vector<ScalarVar> &value);
  OpParamDef &OutputShapeDependOnCompute();
  OpParamDef &Follow(const char *paramName);
  OpParamDef &Follow(const char *paramName, FollowType ftype);
  OpParamDef &Comment(const char *comment);

private:
  friend class AclnnOpGenerator;
  friend class Generator;
  friend class OpProtoGenerator;
  friend class GeneratorFactory;
  friend class CfgGenerator;
  friend class OpParamTrunk;
  friend class OpDef;
  friend class OpDefUtils;

  bool operator==(const OpParamDef &def) const;
  void MergeParam(const OpParamDef &def);
  ge::AscendString &GetParamName(void) const;
  Option GetParamType(void);
  std::vector<ge::DataType> &GetDataTypes(void);
  std::vector<ge::DataType> &GetOriginDataTypes(void);
  std::vector<ge::DataType> &GetDataTypesList(void);
  std::vector<ge::Format> &GetFormats(void);
  std::vector<ge::Format> &GetFormatsList(void);
  std::vector<ge::Format> &GetUnknownShapeFormats(void);
  ge::AscendString &GetValueDepend(void) const;
  DependScope &GetDependScope(void) const;
  ge::AscendString &GetFollowName(void) const;
  FollowType &GetFollowType(void) const;
  ge::AscendString &GetComment(void) const;
  bool GetIgnoreContiguous(void);
  bool GetAutoContiguous(void);
  bool IsScalar(void) const;
  bool IsScalarList(void) const;
  bool IsScalarOrScalarList(void) const;
  bool IsScalarTypeSet(void) const;
  bool IsScalarNameSet(void) const;
  bool IsValueDepend(void) const;
  bool IsDtype(void) const;
  bool IsDtypeList(void) const;
  bool IsFormat(void) const;
  bool IsFormatList(void) const;
  bool IsOutputShapeDependOnCompute(void) const;
  ge::AscendString &GetScalarName(void) const;
  ge::DataType GetScalarType(void) const;
  uint32_t GetVersion(void);
  InitValueType &GetInitValueType(void);
  InitValueNum &GetInitValue(void);
  std::vector<ScalarVar> &GetInitValueList(void);
  std::unique_ptr<OpParamDefImpl> impl_;
};

class OpAttrDefImpl;
class OpAttrDef {
public:
  explicit OpAttrDef(const char *name);
  OpAttrDef(const OpAttrDef &attr_def);
  ~OpAttrDef();
  OpAttrDef &operator=(const OpAttrDef &attr_def);
  OpAttrDef &AttrType(Option attr_type);
  OpAttrDef &Bool(void);
  OpAttrDef &Bool(bool value);
  OpAttrDef &Float(void);
  OpAttrDef &Float(float value);
  OpAttrDef &Int(void);
  OpAttrDef &Int(int64_t value);
  OpAttrDef &String(void);
  OpAttrDef &String(const char *value);
  OpAttrDef &ListBool(void);
  OpAttrDef &ListBool(std::vector<bool> value);
  OpAttrDef &ListFloat(void);
  OpAttrDef &ListFloat(std::vector<float> value);
  OpAttrDef &ListInt(void);
  OpAttrDef &ListInt(std::vector<int64_t> value);
  OpAttrDef &ListListInt(void);
  OpAttrDef &ListListInt(std::vector<std::vector<int64_t>> value);
  OpAttrDef &Version(uint32_t version);
  OpAttrDef &Comment(const char *comment);
  ge::AscendString &GetName(void) const;
  bool IsRequired(void);

private:
  friend class AclnnOpGenerator;
  friend class Generator;
  friend class OpProtoGenerator;
  friend class GeneratorFactory;
  friend class CfgGenerator;
  friend class OpParamTrunk;
  friend class OpDef;
  friend class OpDefUtils;

  bool operator==(const OpAttrDef &attr_def) const;
  ge::AscendString &GetCfgDataType(void) const;
  ge::AscendString &GetProtoDataType(void) const;
  ge::AscendString &GetAttrDefaultVal(const char *brac);
  uint32_t GetVersion(void);
  ge::AscendString &GetComment(void) const;

  std::unique_ptr<OpAttrDefImpl> impl_;
};

class OpAICoreConfigImpl;
class OpAICoreConfig {
public:
  OpAICoreConfig();
  OpAICoreConfig(const OpAICoreConfig &aicore_config);
  ~OpAICoreConfig();
  OpAICoreConfig &operator=(const OpAICoreConfig &aicore_config);
  OpParamDef &Input(const char *name);
  OpParamDef &Output(const char *name);
  OpAICoreConfig &DynamicCompileStaticFlag(bool flag);
  OpAICoreConfig &DynamicFormatFlag(bool flag);
  OpAICoreConfig &DynamicRankSupportFlag(bool flag);
  OpAICoreConfig &DynamicShapeSupportFlag(bool flag);
  OpAICoreConfig &NeedCheckSupportFlag(bool flag);
  OpAICoreConfig &PrecisionReduceFlag(bool flag);
  OpAICoreConfig &ExtendCfgInfo(const char *key, const char *value);

private:
  friend class AclnnOpGenerator;
  friend class Generator;
  friend class OpProtoGenerator;
  friend class GeneratorFactory;
  friend class CfgGenerator;
  friend class OpParamTrunk;
  friend class OpDef;
  friend class OpDefUtils;

  std::vector<OpParamDef> &GetInputs(void) const;
  std::vector<OpParamDef> &GetOutputs(void) const;
  std::vector<ge::AscendString> &GetCfgKeys(void);
  std::map<ge::AscendString, ge::AscendString> &GetCfgInfo(void);
  ge::AscendString &GetConfigValue(const char *key);
  void AddCfgItem(const char *key, const char *value);

  std::unique_ptr<OpAICoreConfigImpl> impl_;
};

class OpAICoreDefImpl;
class OpAICoreDef {
public:
  OpAICoreDef();
  OpAICoreDef(const OpAICoreDef &aicore_def);
  ~OpAICoreDef();
  OpAICoreDef &operator=(const OpAICoreDef &aicore_def);
  OpAICoreDef &SetTiling(gert::OpImplRegisterV2::TilingKernelFunc func);
  OpAICoreDef &SetCheckSupport(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetOpSelectFormat(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetOpSupportInfo(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetOpSpecInfo(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetParamGeneralize(optiling::PARAM_GENERALIZE_FUNC func);
  gert::OpImplRegisterV2::TilingKernelFunc &GetTiling(void);
  optiling::OP_CHECK_FUNC &GetCheckSupport(void);
  optiling::OP_CHECK_FUNC &GetOpSelectFormat(void);
  optiling::OP_CHECK_FUNC &GetOpSupportInfo(void);
  optiling::OP_CHECK_FUNC &GetOpSpecInfo(void);
  optiling::PARAM_GENERALIZE_FUNC &GetParamGeneralize(void);
  OpAICoreDef &AddConfig(const char *soc);
  OpAICoreDef &AddConfig(const char *soc, OpAICoreConfig &aicore_config);

private:
  friend class AclnnOpGenerator;
  friend class Generator;
  friend class OpProtoGenerator;
  friend class GeneratorFactory;
  friend class CfgGenerator;
  friend class OpParamTrunk;
  friend class OpDef;
  friend class OpDefUtils;

  std::map<ge::AscendString, OpAICoreConfig> &GetAICoreConfigs(void);
  void Log(const char *op_type, const char *info) const;

  std::unique_ptr<OpAICoreDefImpl> impl_;
};

class OpMC2DefImpl;
class OpMC2Def {
public:
  OpMC2Def();
  OpMC2Def(const OpMC2Def &mc2_def);
  ~OpMC2Def();
  OpMC2Def &operator=(const OpMC2Def &mc2_def);
  OpMC2Def &HcclGroup(const char *value);
  OpMC2Def &HcclGroup(std::vector<const char *> value);

private:
  friend class AclnnOpGenerator;
  friend class Generator;
  friend class OpProtoGenerator;
  friend class GeneratorFactory;
  friend class CfgGenerator;
  friend class OpParamTrunk;

  std::vector<ge::AscendString> &GetHcclGroups(void) const;
  std::unique_ptr<OpMC2DefImpl> impl_;
};

class OpDefImpl;
class OpDef {
public:
  explicit OpDef(const char *type);
  OpDef(const OpDef &op_def);
  ~OpDef();
  OpDef &operator=(const OpDef &op_def);
  OpParamDef &Input(const char *name);
  OpParamDef &Output(const char *name);
  OpAttrDef &Attr(const char *name);
  OpDef &Comment(CommentSection section, const char *comment);
  OpDef &SetInferShape(gert::OpImplRegisterV2::InferShapeKernelFunc func);
  OpDef &SetInferShapeRange(gert::OpImplRegisterV2::InferShapeRangeKernelFunc func);
  OpDef &SetInferDataType(gert::OpImplRegisterV2::InferDataTypeKernelFunc func);
  gert::OpImplRegisterV2::InferShapeKernelFunc &GetInferShape(void);
  gert::OpImplRegisterV2::InferShapeRangeKernelFunc &GetInferShapeRange(void);
  gert::OpImplRegisterV2::InferDataTypeKernelFunc &GetInferDataType(void);
  OpAICoreDef &AICore(void);
  OpMC2Def &MC2(void);

private:
  friend class AclnnOpGenerator;
  friend class Generator;
  friend class OpProtoGenerator;
  friend class GeneratorFactory;
  friend class CfgGenerator;
  friend class OpParamTrunk;
  friend class OpDefUtils;
  struct DfsParam {
    std::vector<std::vector<ge::DataType>> full_types;
    std::vector<std::vector<ge::Format>> full_formats;
    std::vector<ge::DataType> types;
    std::vector<ge::Format> formats;
  };
  enum class PortStat : uint32_t {
    IN = 0,
    OUT = 1,
    INOUT = 2,
    INVALID_STAT
  };
  struct PortFollowInfo {
    PortStat port_stat = PortStat::IN;
    uint32_t index_in = 0;
    uint32_t index_out = 0;
    ge::AscendString follow_port_name = "";
    FollowType follow_type = FollowType::ALL;
  };
  ge::AscendString &GetOpType(void);
  ge::AscendString &GetCateGory(void) const;
  std::vector<ge::AscendString> &GetBrief(void) const;
  std::vector<ge::AscendString> &GetConstraints(void) const;
  std::vector<ge::AscendString> &GetRestrictions(void) const;
  std::vector<ge::AscendString> &GetSee(void) const;
  std::vector<ge::AscendString> &GetThirdPartyFwkCopat(void) const;
  std::vector<OpParamDef> &GetInputs(void);
  std::vector<OpParamDef> &GetOutputs(void);
  std::vector<OpAttrDef> &GetAttrs(void);
  std::vector<OpParamDef> GetMergeInputs(OpAICoreConfig &aicore_config);
  std::vector<OpParamDef> GetMergeOutputs(OpAICoreConfig &aicore_config);
  void CheckIncompatible(const std::vector<OpParamDef>& all) const;
  void FullPermutation(const std::vector<OpParamDef> &input_param, const std::vector<OpParamDef> &output_param) const;
  void DfsFullPermutation(DfsParam &dfs_param, const std::vector<OpParamDef> &all_param,
                          uint32_t list_idx, uint32_t non_list_idx) const;
  void DfsDataType(DfsParam &dfs_param, const std::vector<OpParamDef> &all_param,
                   uint32_t list_idx, uint32_t non_list_idx) const;
  void DfsFormat(DfsParam &dfs_param, const std::vector<OpParamDef> &all_param,
                 uint32_t list_idx, uint32_t non_list_idx) const;
  uint32_t CalcNonListLen(const std::vector<OpParamDef> &all_param) const;
  bool IsNonListTypes(const OpParamDef &def) const;
  bool IsNonListFormats(const OpParamDef &def) const;
  void SetDefaultND(const std::vector<OpParamDef> &defs) const;
  std::vector<std::vector<OpParamDef>> GetMergeInputsOutputs(const OpAICoreConfig &aicore_config);
  void SetPermutedInputOutput(const DfsParam &dfs_param, const std::vector<OpParamDef> &input,
                              const std::vector<OpParamDef> &output) const;
  void MergeParam(std::vector<OpParamDef> &merge, std::vector<OpParamDef> &aicore_params) const;
  ItemFindStatus FindAttr(const char *name, OpAttrDef **attr);
  OpAttrDef &AddAttr(OpAttrDef &attr);
  OpAttrDef &GetOrCreateAttr(const char *name);
  void FollowImpl(void);
  void FollowListImpl(std::vector<OpParamDef>& input, std::vector<OpParamDef>& output);
  std::map<ge::AscendString, OpDef::PortFollowInfo> GetFollowMap(void);
  std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> GetFollowShapeMap(void);
  std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> GetFollowTypeMap(void);
  OpParamDef GetParamDef(const ge::AscendString& name, OpDef::PortStat stat);

  std::unique_ptr<OpDefImpl> impl_;
};
}  // namespace ops

#endif
