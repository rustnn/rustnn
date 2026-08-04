/**
 * CANN Adapter Layer - Internal Implementation Structures
 *
 * This header defines the internal struct layouts used by the adapter's
 * .cc files. It is NOT part of the public C API -- include only in .cc files.
 *
 * DO NOT include this header from any public .h file.
 */

#ifndef CANN_ADAPTER_INTERNAL_H
#define CANN_ADAPTER_INTERNAL_H

#include "adapter_types.h"

#include "compatible/AiTensor.h"
#include "compatible/HiAiModelBuilderType.h"
#include "compatible/hiai_base_types_cpt.h"
#include "HiAiModelManagerService.h"
#include "HiAiModelManagerType.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/tensor.h"
#include "graph/shape.h"
#include "hiai_ir_build.h"

namespace ddk {

/* ── CannShapeImpl (wraps ge::Shape) ───────────────────────────────── */

struct CannShapeImpl {
    ge::Shape shape;
    CannShapeImpl() : shape() {}
    explicit CannShapeImpl(const std::vector<int64_t>& v) : shape(v) {}
    explicit CannShapeImpl(const ge::Shape& s) : shape(s) {}
};

/* ── CannGraphImpl ─────────────────────────────────────────────────── */

struct CannGraphImpl {
    ge::Graph graph;
    explicit CannGraphImpl(const std::string& name) : graph(name) {}
    CannGraphImpl() : graph("") {}
};

/* ── CannOpTensorDescImpl ──────────────────────────────────────────────── */

struct CannOpTensorDescImpl {
    ge::TensorDesc desc;
};

/* ── CannOpTensorDescImpl ──────────────────────────────────────────────── */
struct CannOpTensorImpl {
    ge::Tensor tensor;
    explicit CannOpTensorImpl(const ge::TensorDesc& desc) : tensor(desc) {}
};

/* ── CannIOTensorImpl (wraps hiai::AiTensor) ─────────────────────────── */

struct CannIOTensorImpl {
    std::shared_ptr<hiai::AiTensor> tensor;
    hiai::TensorDimension dim;
    explicit CannIOTensorImpl() : tensor(std::make_shared<hiai::AiTensor>()) {}
};

/* ── CannIOTensorDimImpl (wraps hiai::TensorDimension) ───────────────── */

struct CannIOTensorDimImpl {
    hiai::TensorDimension dim;
    explicit CannIOTensorDimImpl(const hiai::TensorDimension& d) : dim(d) {}
    CannIOTensorDimImpl(uint32_t n, uint32_t c, uint32_t h, uint32_t w)
        : dim(n, c, h, w) {}
    explicit CannIOTensorDimImpl(const std::vector<uint32_t>& v) : dim(v) {}
    CannIOTensorDimImpl() = default;
};

/* ── CannHiaiIrBuildImpl ─────────────────────────────────────────────────── */

struct CannHiaiIrBuildImpl {
    hiai::HiaiIrBuild build;
};

/* ── CannModelImpl ─────────────────────────────────────────────────── */

struct CannModelImpl {
    ge::Model model;
        CannModelImpl() : model() {}
    explicit CannModelImpl(const std::string& name) : model(name, "") {}
};

/* ── CannBuildOptsImpl ──────────────────────────────────────────────── */

struct CannBuildOptsImpl {
    hiai::BuildOptions options;
};

/* ── CannContextImpl ────────────────────────────────────────────────── */

struct CannContextImpl {
    hiai::AiContext context;
};

/* ── CannModelDescImpl ──────────────────────────────────────────────── */

struct CannModelDescImpl {
    std::shared_ptr<hiai::AiModelDescription> desc;
    CannModelDescImpl(const std::string& name, int32_t freq, int32_t fw,
                      int32_t mtype, int32_t dtype)
        : desc(std::make_shared<hiai::AiModelDescription>(name, freq, fw, mtype, dtype)) {}
};

/* ── CannModelMgrImpl ─────────────────────────────────────────────── */

struct CannModelMgrImpl {
    hiai::AiModelMngerClient client;
};

/* ── CannOperatorImpl ─────────────────────────────────────────────── */
//TODO: See if we want to simulate the same API as sdk.
// Consideration: would there be any changes in the DDK API?
// struct CannOperatorImpl {
//     ge::Operator op;
//     explicit CannOperatorImpl(const std::string& name, std::string(type)) {
//         op = ge::Operator(name, type);
//     }
// };

/* ── Utility inline converters ───────────────────────────────────────── */


inline ge::Operator* ToGeOp(CannOperatorHandle op) {
    return reinterpret_cast<ge::Operator*>(op);
}

inline CannOperatorHandle FromGeOp(ge::Operator* op) {
    return reinterpret_cast<CannOperatorHandle>(op);
}


}  // namespace ddk

#endif /* CANN_ADAPTER_INTERNAL_H */
