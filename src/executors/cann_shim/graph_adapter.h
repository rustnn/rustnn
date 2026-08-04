/**
 * CANN Adapter Layer - Graph Wrapper
 *
 * Wraps ge::Graph with a pure C interface.
 */

#ifndef CANN_GRAPH_ADAPTER_H
#define CANN_GRAPH_ADAPTER_H

#include "adapter_types.h"

namespace ddk {
extern "C" {

/* ── Graph lifecycle ─────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannGraphHandle cann_graph_create(const char* name);
CANN_ADAPTER_EXPORT void            cann_graph_destroy(CannGraphHandle graph);

/* ── Operator management ─────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_graph_add_op(CannGraphHandle graph, CannOperatorHandle op);

/* ── Input/output setting ────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_graph_set_inputs(CannGraphHandle graph,
                                  CannOperatorHandle* inputs,
                                  int32_t input_count);

CANN_ADAPTER_EXPORT CannStatus cann_graph_set_outputs(CannGraphHandle graph,
                                   CannOperatorHandle* outputs,
                                   int32_t output_count);

/* ── Query ───────────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_graph_find_op_by_name(CannGraphHandle graph, const char* name);
CANN_ADAPTER_EXPORT int32_t            cann_graph_is_valid(CannGraphHandle graph);

}  // extern "C"
}  // namespace ddk

#endif /* CANN_GRAPH_ADAPTER_H */
