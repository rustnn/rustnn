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
#ifndef GE_PUBLIC_TYPES_H
#define GE_PUBLIC_TYPES_H

#include <memory>
#include <vector>

namespace ge {

#if(defined(HOST_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif
#if(defined(DEV_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif

using char_t = char;
using float32_t = float;
using float64_t = double;
using vector_bit_t = std::vector<bool>;

extern const std::vector<int64_t> UNKNOWN_SHAPE;
extern const std::vector<int64_t> UNKNOWN_RANK;

enum DataType {
    DT_UNDEFINED = 17, // Used to indicate a DataType field has not been set.
    DT_FLOAT = 0, // float type
    DT_FLOAT16 = 1, // fp16 type
    DT_INT8 = 2, // int8 type
    DT_INT16 = 6, // int16 type
    DT_UINT16 = 7, // uint16 type
    DT_UINT8 = 4, // uint8 type
    DT_INT32 = 3,
    DT_INT64 = 9, // int64 type
    DT_UINT32 = 8, // unsigned int32
    DT_UINT64 = 10, // unsigned int64
    DT_BOOL = 12, // bool type
    DT_DOUBLE = 11, // double type
    DT_DUAL = 13, /* dual output type */
    DT_DUAL_SUB_INT8 = 14, /* dual output int8 type */
    DT_DUAL_SUB_UINT8 = 15, /* dual output uint8 type */
    DT_COMPLEX64 = 16,
    DT_2BIT = 21, // 2BIT type
    DT_INT4 = 22, // int4 type
    DT_QUINT8 = 23, // quint8 for ann
    DT_RESOURCE = 24, // dataflow resource flow
    DT_3BIT = 25, // 3BIT type
    DT_UINT2 = 26, // uint2 type
    DT_UINT4 = 27, // uint4 type
    DT_STRING = 28, // string type
    DT_FLOAT8_E5M2 = 35,
    DT_FLOAT4_E2M1 = 40,
    DT_MAX
};

int64_t GetSizeInBytes(int64_t element_count, DataType data_type);

enum Format {
    FORMAT_NCHW = 0, /* NCHW */
    FORMAT_NHWC, /* NHWC */
    FORMAT_ND, /* Nd Tensor */
    FORMAT_NC1HWC0, /* NC1HWC0 */
    FORMAT_FRACTAL_Z, /* FRACTAL_Z */
    FORMAT_NC1C0HWPAD,
    FORMAT_NHWC1C0,
    FORMAT_FSR_NCHW,
    FORMAT_FRACTAL_DECONV,
    FORMAT_C1HWNC0,
    FORMAT_FRACTAL_DECONV_TRANSPOSE,
    FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
    FORMAT_NC1HWC0_C04, /* NC1HWC0, C0 =4 */
    FORMAT_FRACTAL_Z_C04, /* FRACZ格式，C0 =4 */
    FORMAT_CHWN,
    FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS,
    FORMAT_HWCN,
    FORMAT_NC1KHKWHWC0, /* KH,KW kernel h& kernel w maxpooling max output format */
    FORMAT_BN_WEIGHT,
    FORMAT_FILTER_HWCK, /* filter input tensor format */
    FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20,
    FORMAT_HASHTABLE_LOOKUP_KEYS,
    FORMAT_HASHTABLE_LOOKUP_VALUE,
    FORMAT_HASHTABLE_LOOKUP_OUTPUT,
    FORMAT_HASHTABLE_LOOKUP_HITS = 24,
    FORMAT_FRACTAL_Z_SPARSE,
    FORMAT_WINO = 27,
    FORMAT_NC4HW4, /* For CPUCL Kernel */
    FORMAT_FRACTAL_Z_N_TRANS,
    FORMAT_FRACTAL_Z_N_NORMAL,
    FORMAT_NC8HW8, /* For CPUCL fp16 Kernel */
    FORMAT_NCDHW = 40,
    FORMAT_NDHWC,
    FORMAT_NDC1HWC0,
    FORMAT_DHWCN,
    FORMAT_FRACTAL_Z_5D,
    FORMAT_FRACTAL_Z_DECONV_SP_K2S2P0,
    FORMAT_FRACTAL_Z_DECONV_SP_K4S2P0,
    FORMAT_FRACTAL_Z_DECONV_SP_K4S2P1,
    FORMAT_FRACTAL_Z_DECONV_SP_K4S2P2,
    FORMAT_WINO_V2,
    FORMAT_FRACTAL_Z_S16, /* for S16S8 filter */
    FORMAT_WINO_V2_S16, /* for wino S16S8 filter */
    FORMAT_FRACTAL_Z_DW_DIAG, // for dw diagonal
    FORMAT_FRACTAL_Z_DW_V2, // for dw v2 kernel
    FRACTAL_NZ, /* L0C format */
    FORMAT_FRACTAL_Z_DECONV_SP_K3S3P0,
    FORMAT_FRACTAL_Z_DECONV_SP_K2S2P0_S16S8,
    FORMAT_FRACTAL_Z_DECONV_SP_K4S2P0_S16S8,
    FORMAT_FRACTAL_Z_DECONV_SP_K4S2P1_S16S8,
    FORMAT_FRACTAL_Z_DECONV_SP_K4S2P2_S16S8,
    FORMAT_FRACTAL_Z_DECONV_SP_K3S3P0_S16S8,
    FORMAT_NC1HWC0_COMPACT, // for compact feature map
    FORMAT_FRACTAL_Z_COMPACT, // for compact format filter, in and out
    FORMAT_FRACTAL_Z_COMPACT_IN, // for compact_in conv format filter
    FORMAT_FRACTAL_Z_COMPACT_OUT, // for compact_out conv format filter
    FORMAT_FRACTAL_DECONV_S16, // deconv s16s8 filter
    FORMAT_FRACTAL_Z_DW_DIAG_S16, // for s16s8 dw diagonal
    FORMAT_FRACTAL_Z_DW_V2_S16, // for s16s8 dw v2 kernel
    FORMAT_FRACTAL_Z_MATMUL_QUANT_S8, // matmul nd filter
    FORMAT_FRACTAL_ZZ_MATMUL_QUANT_S16S8, // matmul nd S16S8 zZ filter
    FORMAT_FRACTAL_NN_MATMUL_QUANT_S16S8, // matmul nd S16S8 nN filter
    FORMAT_FRACTAL_NN, // matmul stitch s16s8 filter
    FORMAT_FRACTAL_NZN, // matmul stitch s16s4 filter
    FRACTAL_NZ_128, // matmul stitch s16s4 deq
    FORMAT_FRACTAL_Z_S16_COMPACT, // for S16S8 filter compact arrangement
    FORMAT_FRACTAL_DECONV_S16_COMPACT, // deconv s16s8 filter compact arrangement
    FORMAT_FRACTAL_Z_CUT_KERNEL_H, // only used in conv_to_l1 s8 cut filterH kernel
    FORMAT_FRACTAL_Z_CUT_KERNEL_H_S16, // only used in conv_to_l1 s16s8 cut filterH kernel
    FORMAT_FRACTAL_Z_S16_C04, // for S16S8 filter C04 arrangement
    FORMAT_RESERVED
};

enum DeviceType {
    NPU = 0,
    CPU = 1,
    GPU = 2,
    ISP = 3,
    DEVICE_TYPE_RESERVED,
};

enum WeightDataType {
    FP32,
    FP16
};

enum AnchorStatus {
    ANCHOR_SUSPEND = 0, // dat null
    ANCHOR_CONST = 1,
    ANCHOR_DATA = 2, // 有效的
    ANCHOR_RESERVED = 3
};

enum PrecisionMode {
    PRECISION_MODE_FP32 = 0, // 设置精度配置选项为FP32，当前对CPUCL生效
    PRECISION_MODE_PREFER_FP16, // 设置精度配置选项为FP16，当算子不支持时，回退到FP32计算，当前对CPUCL生效
};

enum class TuningStrategy {
    OFF = 0,
    ON_DEVICE_TUNING,
    ON_DEVICE_PREPROCESS_TUNING,
    ON_CLOUD_TUNING
};

enum class ExecuteDeviceConfigMode {
    AUTO,
    MODEL_LEVEL,
    OP_LEVEL
};

enum class ExecuteFallBackMode {
    ENABLE,
    DISABLE
};

enum class DeviceMemoryReusePlan {
    UNSET,
    LOW,
    HIGH
};

struct TensorType {
    explicit TensorType(DataType dt)
    {
        dtVec.push_back(dt);
    };

    TensorType(std::initializer_list<DataType> types)
    {
        dtVec = types;
    };

    static TensorType ALL()
    {
        return TensorType{DT_BOOL, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_QUINT8,
            DT_RESOURCE, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
    }
    static TensorType QuantifiedType() { return TensorType{DT_QUINT8}; }

    static TensorType OrdinaryType()
    {
        return TensorType{DT_QUINT8};
    }

    static TensorType BasicType()
    {
        return TensorType{DT_BOOL, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_QUINT8,
            DT_RESOURCE, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
    }

    static TensorType NumberType()
    {
        return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
    }

    static TensorType RealNumberType()
    {
        return TensorType{DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8,
            DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
    }

    static TensorType IntegerDataType()
    {
        return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
    }

    static TensorType SignedDataType() { return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8}; }

    static TensorType UnsignedDataType() { return TensorType{DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}; }

    static TensorType FloatingDataType() { return TensorType{DT_DOUBLE, DT_FLOAT, DT_FLOAT16}; }

    static TensorType IndexNumberType() { return TensorType{DT_INT32, DT_INT64}; }

    static TensorType UnaryDataType()
    {
        return TensorType{DT_DOUBLE, DT_FLOAT, DT_FLOAT16};
    }

    static TensorType FLOAT() { return TensorType{DT_FLOAT, DT_FLOAT16}; }
    std::vector<DataType> dtVec;
};
} // namespace ge

namespace domi {
enum class ImplyType : unsigned int {
  BUILDIN = 0,  // Built in operator, normally executed by OME
  TVM,          // Compile to TVM bin file for execution
  CUSTOM,       // User defined calculation logic, executed by CPU
  AI_CPU,       // AICPU
  CCE,          // Cce
  GELOCAL,      // GE local, do node need execute by device
  HCCL,         // Hccl
  INVALID = 0xFFFFFFFF,
};
using char_t = ge::char_t;
using float32_t = ge::float32_t;
using float64_t = ge::float64_t;
}  // namespace domi

#endif // GE_PUBLIC_TYPES_H
