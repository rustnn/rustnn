/**
 * CANN Adapter Layer - Shared C Types
 *
 * This header defines C-compatible types that serve as the public interface
 * between the Chromium WebNN CANN backend and the C++ CANN libraries.
 *
 * All types are pure C (no C++ features) to avoid ABI incompatibility
 * between different libc++ versions.
 */

#ifndef CANN_ADAPTER_TYPES_H
#define CANN_ADAPTER_TYPES_H

#include <stdint.h>

namespace ddk {

#if defined(_WIN32)
  #define CANN_ADAPTER_EXPORT __declspec(dllexport)
#else
  #define CANN_ADAPTER_EXPORT __attribute__((visibility("default")))
#endif

/* ── Opaque handles ────────────────────────────────────────────────────── */
/* These hide all C++ objects from the public interface. */

typedef struct CannGraphImpl*           CannGraphHandle;
typedef struct CannOperatorImpl*       CannOperatorHandle;
typedef struct CannOpTensorDescImpl*     CannOpTensorDescHandle;
typedef struct CannOpTensorImpl*        CannOpTensorHandle;
typedef struct CannIOTensorImpl*         CannIOTensorHandle;
typedef struct CannIOTensorDimImpl*      CannIOTensorDimensionHandle;
typedef struct CannModelDescImpl*      CannModelDescHandle;
typedef struct CannModelImpl*          CannModelHandle;
typedef struct CannBuildOptsImpl*      CannBuildOptionsHandle;
typedef struct CannModelMgrImpl*       CannModelManagerHandle;
typedef struct CannShapeImpl*          CannShapeHandle;
typedef struct CannContextImpl*        CannContextHandle;
typedef struct CannHiaiIrBuildImpl*     CannHiaiIrBuildHandle;

/* ── Status codes ───────────────────────────────────────────────────────── */

using CannStatus = int32_t;

// Scoped constants instead of macros
constexpr CannStatus kSuccess        = 0;
constexpr CannStatus kFailed         = 1;
constexpr CannStatus kNotInit        = 2;
constexpr CannStatus kInvalidPara    = 3;
constexpr CannStatus kInvalidApi     = 7;
constexpr CannStatus kInvalidPtr     = 8;

/* ── Data type enumeration (mirrors ge::DataType) ─────────────────────── */

typedef enum {
    CANN_DT_UNDEFINED      = 17,
    CANN_DT_FLOAT          = 0,
    CANN_DT_FLOAT16        = 1,
    CANN_DT_INT8           = 2,
    CANN_DT_INT32          = 3,
    CANN_DT_UINT8          = 4,
    CANN_DT_INT16          = 6,
    CANN_DT_UINT16         = 7,
    CANN_DT_UINT32         = 8,
    CANN_DT_INT64          = 9,
    CANN_DT_UINT64         = 10,
    CANN_DT_DOUBLE         = 11,
    CANN_DT_BOOL           = 12,
    CANN_DT_DUAL           = 13,
    CANN_DT_DUAL_SUB_INT8  = 14,
    CANN_DT_DUAL_SUB_UINT8 = 15,
    CANN_DT_COMPLEX64      = 16,
    CANN_DT_2BIT           = 21,
    CANN_DT_INT4           = 22,
    CANN_DT_QUINT8         = 23,
    CANN_DT_RESOURCE       = 24,
    CANN_DT_3BIT           = 25,
    CANN_DT_UINT2          = 26,
    CANN_DT_UINT4          = 27,
    CANN_DT_STRING         = 28,
    CANN_DT_FLOAT8_E5M2    = 35,
    CANN_DT_FLOAT4_E2M1    = 40,
    CANN_DT_MAX            = 41
} CannDataType;

/* ── Format enumeration (mirrors ge::Format, essential subset) ────────── */

typedef enum {
    CANN_FORMAT_NCHW    = 0,
    CANN_FORMAT_NHWC    = 1,
    CANN_FORMAT_ND      = 2,
    CANN_FORMAT_NC1HWC0 = 3,
    CANN_FORMAT_FRACTAL_Z = 4,
    CANN_FORMAT_RESERVED = 88
} CannFormat;

/* ── Model buffer (C-compatible, mirrors hiai::ModelBufferData) ───────── */

typedef struct {
    void*    data;
    uint32_t length;
} CannModelBuffer;

/* ── Activation mode (for activation operator) ────────────────────────── */

typedef enum {
    CANN_ACTIVATION_SIGMOID  = 0,
    CANN_ACTIVATION_RELU     = 1,
    CANN_ACTIVATION_TANH     = 2,
    CANN_ACTIVATION_GELU     = 3,
    CANN_ACTIVATION_LEAKY_RELU = 5,
    CANN_ACTIVATION_ELU      = 6,
    CANN_ACTIVATION_SELU     = 7,
    CANN_ACTIVATION_SOFTPLUS = 8,
    CANN_ACTIVATION_SOFTSIGN = 9,
    CANN_ACTIVATION_HSIGMOID = 10,
    CANN_ACTIVATION_RELU6    = 12,
    CANN_ACTIVATION_ABS      = 13,
    CANN_ACTIVATION_SWISH    = 14
} CannActivationMode;

/* ── Pooling mode ─────────────────────────────────────────────────────── */

typedef enum {
    CANN_POOL_AVG = 0,
    CANN_POOL_MAX = 1
} CannPoolMode;

/* ── Model description constants (mirrors hiai enums) ─────────────────── */

typedef enum {
    CANN_MODEL_FREQ_LOW    = 1,
    CANN_MODEL_FREQ_MEDIUM = 2,
    CANN_MODEL_FREQ_HIGH   = 3,
    CANN_MODEL_FREQ_EXTREME = 4,
    CANN_MODEL_FREQ_HIGH_COMPUTE_UNIT = 103,
    CANN_MODEL_FREQ_MEDIUM_BAND_MEDIUM = 202
} CannModelFrequency;

typedef enum {
    CANN_MODEL_DEVICE_NPU = 0,
    CANN_MODEL_DEVICE_IPU = 1,
    CANN_MODEL_DEVICE_CPU = 3
} CannModelDeviceType;

typedef enum {
    CANN_MODEL_FRAMEWORK_NONE       = 0,
    CANN_MODEL_FRAMEWORK_TENSORFLOW = 1,
    CANN_MODEL_FRAMEWORK_KALDI      = 2,
    CANN_MODEL_FRAMEWORK_CAFFE      = 3,
    CANN_MODEL_FRAMEWORK_TF_8BIT    = 4,
    CANN_MODEL_FRAMEWORK_CAFFE_8BIT = 5
} CannModelFramework;

typedef enum {
    CANN_MODEL_TYPE_ONLINE  = 0,
    CANN_MODEL_TYPE_OFFLINE = 1
} CannModelType;

/* ── Build option enums ───────────────────────────────────────────────── */

typedef enum {
    CANN_BUILD_MODE_AUTO   = 0,
    CANN_BUILD_MODE_CUSTOM = 1
} CannBuildMode;

typedef enum {
    CANN_WEIGHT_DT_FP32 = 0,
    CANN_WEIGHT_DT_FP16 = 1
} CannWeightDataType;

typedef enum {
    CANN_PRECISION_FP32       = 0,
    CANN_PRECISION_PREFER_FP16 = 1
} CannPrecisionMode;

}  // namespace ddk

#endif /* CANN_ADAPTER_TYPES_H */
