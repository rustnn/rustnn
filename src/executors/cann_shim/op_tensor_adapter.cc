/**
 * CANN Adapter Layer - OpTensor & TensorDesc Implementation
 */

#include "op_tensor_adapter.h"
#include "adapter_internal.h"

#include <cstring>
#include <vector>

#include "graph/types.h"

namespace ddk {
extern "C" {

/* ── Internal helpers: value mapping ───────────────────────────────────── */

static ge::DataType ToGeDataType(CannDataType t) {
    return static_cast<ge::DataType>(t);
}

static CannDataType FromGeDataType(ge::DataType t) {
    return static_cast<CannDataType>(t);
}

static ge::Format ToGeFormat(CannFormat f) {
    return static_cast<ge::Format>(f);
}

static CannFormat FromGeFormat(ge::Format f) {
    return static_cast<CannFormat>(f);
}


/* ── TensorDesc lifecycle ──────────────────────────────────────────────── */

CannOpTensorDescHandle cann_tensor_desc_create(CannShapeHandle shape,
                                              CannFormat format,
                                              CannDataType dtype) {
    if (!shape) return nullptr;
    ge::Shape geShape(shape->shape);

    return new CannOpTensorDescImpl{ge::TensorDesc(geShape, ToGeFormat(format), ToGeDataType(dtype))};
}

void cann_tensor_desc_destroy(CannOpTensorDescHandle desc) {
    delete desc;
}

/* ── TensorDesc property setters ───────────────────────────────────────── */

CannStatus cann_tensor_desc_set_shape(CannOpTensorDescHandle desc,
                                       const int64_t* shape,
                                       int32_t shape_count) {
    if (!desc || !shape || shape_count <= 0) return kInvalidPara;
    std::vector<int64_t> s(shape, shape + shape_count);
    desc->desc.SetShape(ge::Shape(s));
    return kSuccess;
}

CannStatus cann_tensor_desc_set_format(CannOpTensorDescHandle desc, CannFormat format) {
    if (!desc) return kInvalidPtr;
    desc->desc.SetFormat(ToGeFormat(format));
    return kSuccess;
}

CannStatus cann_tensor_desc_set_data_type(CannOpTensorDescHandle desc, CannDataType dtype) {
    if (!desc) return kInvalidPtr;
    desc->desc.SetDataType(ToGeDataType(dtype));
    return kSuccess;
}

/* ── TensorDesc property getters ───────────────────────────────────────── */

const int64_t* cann_tensor_desc_get_shape(CannOpTensorDescHandle desc,
                                           int32_t* out_shape_count) {
    if (!desc || !out_shape_count) return nullptr;
    const std::vector<int64_t>& dims = desc->desc.GetShape().GetDims();
    *out_shape_count = static_cast<int32_t>(dims.size());
    return dims.data();
}

CannFormat cann_tensor_desc_get_format(CannOpTensorDescHandle desc) {
    if (!desc) return CANN_FORMAT_RESERVED;
    return FromGeFormat(desc->desc.GetFormat());
}

CannDataType cann_tensor_desc_get_data_type(CannOpTensorDescHandle desc) {
    if (!desc) return CANN_DT_UNDEFINED;
    return FromGeDataType(desc->desc.GetDataType());
}

/* ── TensorDesc advanced accessors ───────────────────────────────────────── */

CannShapeHandle cann_tensor_desc_get_shape_handle(CannOpTensorDescHandle desc) {
    if (!desc) return nullptr;
    return new CannShapeImpl(desc->desc.GetShape());
}

CannStatus cann_tensor_desc_set_shape_from_handle(CannOpTensorDescHandle desc,
                                                   CannShapeHandle shape) {
    if (!desc || !shape) return kInvalidPtr;
    desc->desc.SetShape(shape->shape);
    return kSuccess;
}

CannShapeHandle cann_tensor_desc_get_origin_shape(CannOpTensorDescHandle desc) {
    if (!desc) return nullptr;
    return new CannShapeImpl(desc->desc.GetOriginShape());
}

CannStatus cann_tensor_desc_set_origin_shape(CannOpTensorDescHandle desc,
                                              CannShapeHandle shape) {
    if (!desc || !shape) return kInvalidPtr;
    desc->desc.SetOriginShape(shape->shape);
    return kSuccess;
}

CannFormat cann_tensor_desc_get_origin_format(CannOpTensorDescHandle desc) {
    if (!desc) return CANN_FORMAT_RESERVED;
    return FromGeFormat(desc->desc.GetOriginFormat());
}

CannStatus cann_tensor_desc_set_origin_format(CannOpTensorDescHandle desc,
                                               CannFormat format) {
    if (!desc) return kInvalidPtr;
    desc->desc.SetOriginFormat(ToGeFormat(format));
    return kSuccess;
}

CannDataType cann_tensor_desc_get_origin_data_type(CannOpTensorDescHandle desc) {
    if (!desc) return CANN_DT_UNDEFINED;
    return FromGeDataType(desc->desc.GetOriginDatatype());
}

CannStatus cann_tensor_desc_set_origin_data_type(CannOpTensorDescHandle desc,
                                                  CannDataType dtype) {
    if (!desc) return kInvalidPtr;
    desc->desc.SetOriginDatatype(ToGeDataType(dtype));
    return kSuccess;
}

/* ── Shape lifecycle ────────────────────────────────────────────────────── */

CannShapeHandle cann_shape_create(const int64_t* dims, int32_t dim_count) {
    if (!dims || dim_count <= 0) return nullptr;
    std::vector<int64_t> v(dims, dims + dim_count);
    return new CannShapeImpl(v);
}

CannShapeHandle cann_shape_create_default(void) {
    return new CannShapeImpl();
}

void cann_shape_destroy(CannShapeHandle shape) {
    delete shape;
}

int32_t cann_shape_get_dim_num(CannShapeHandle shape) {
    if (!shape) return 0;
    return static_cast<int32_t>(shape->shape.GetDimNum());
}

int64_t cann_shape_get_dim(CannShapeHandle shape, int32_t idx) {
    if (!shape || idx < 0) return 0;
    return shape->shape.GetDim(static_cast<size_t>(idx));
}

CannStatus cann_shape_set_dim(CannShapeHandle shape, int32_t idx, int64_t value) {
    if (!shape || idx < 0) return kInvalidPara;
    ge::GraphErrCodeStatus ret = shape->shape.SetDim(static_cast<size_t>(idx), value);
    return (ret == ge::GRAPH_SUCCESS) ? kSuccess : kFailed;
}

const int64_t* cann_shape_get_dims(CannShapeHandle shape, int32_t* out_count) {
    if (!shape || !out_count) return nullptr;
    const std::vector<int64_t>& dims = shape->shape.GetDims();
    *out_count = static_cast<int32_t>(dims.size());
    return dims.data();
}

int64_t cann_shape_get_total_dim_num(CannShapeHandle shape) {
    if (!shape) return 0;
    return shape->shape.GetTotalDimNum();
}

uint32_t cann_shape_get_shape_size(CannShapeHandle shape) {
    if (!shape) return 0;
    return shape->shape.GetShapeSize();
}

/* ── Tensor lifecycle ──────────────────────────────────────────────────── */

CannOpTensorHandle cann_op_tensor_create(CannOpTensorDescHandle desc) {
    if (!desc) return nullptr;
    auto* t = new CannOpTensorImpl(desc->desc);
    return reinterpret_cast<CannOpTensorHandle>(t);
}

void cann_op_tensor_destroy(CannOpTensorHandle tensor) {
    delete reinterpret_cast<CannOpTensorImpl*>(tensor);
}

/* ── Tensor data access ────────────────────────────────────────────────── */

CannOpTensorDescHandle cann_op_tensor_get_desc(CannOpTensorHandle tensor) {
    if (!tensor) return nullptr;
    auto* t = reinterpret_cast<CannOpTensorImpl*>(tensor);
    return new CannOpTensorDescImpl{t->tensor.GetTensorDesc()};
}

CannStatus cann_op_tensor_set_data(CannOpTensorHandle tensor,
                                 const void* data,
                                 uint32_t size) {
    if (!tensor || !data) return kInvalidPtr;
    auto* t = reinterpret_cast<CannOpTensorImpl*>(tensor);
    ge::GraphErrCodeStatus ret = t->tensor.SetData(static_cast<const uint8_t*>(data), size);
    if (ret != ge::GRAPH_SUCCESS) return kFailed;

    return kSuccess;
}

}  // extern "C"
}  // namespace ddk