/**
 * CANN Adapter Layer - Graph Wrapper Implementation
 *
 * Wraps ge::Graph with a pure C interface.
 */

#include "graph_adapter.h"
#include "adapter_internal.h"

#include <string>
#include <vector>

#include "operator_adapter.h"

namespace ddk {
extern "C" {

/* ── Graph lifecycle ───────────────────────────────────────────────────── */

CannGraphHandle cann_graph_create(const char* name) {
    if (!name) return nullptr;
    return new CannGraphImpl(std::string(name));
}

void cann_graph_destroy(CannGraphHandle graph) {
    delete graph;
}

/* ── Operator management ───────────────────────────────────────────────── */

CannStatus cann_graph_add_op(CannGraphHandle graph, CannOperatorHandle op) {
    if (!graph || !op) return kInvalidPtr;
    ge::GraphErrCodeStatus ret = graph->graph.AddOp(*ToGeOp(op));
    return (ret == ge::GRAPH_SUCCESS) ? kSuccess : kFailed;
}

/* ── Input/output setting ──────────────────────────────────────────────── */

CannStatus cann_graph_set_inputs(CannGraphHandle graph,
                                  CannOperatorHandle* inputs,
                                  int32_t input_count) {
    if (!graph || !inputs || input_count <= 0) return kInvalidPara;
    std::vector<ge::Operator> geInputs;
    geInputs.reserve(static_cast<size_t>(input_count));
    for (int32_t i = 0; i < input_count; ++i) {
        if (!inputs[i]) return kInvalidPtr;
        geInputs.push_back(*ToGeOp(inputs[i]));
    }
    graph->graph.SetInputs(geInputs);
    return kSuccess;
}

CannStatus cann_graph_set_outputs(CannGraphHandle graph,
                                   CannOperatorHandle* outputs,
                                   int32_t output_count) {
    if (!graph || !outputs || output_count <= 0) return kInvalidPara;
    std::vector<ge::Operator> geOutputs;
    geOutputs.reserve(static_cast<size_t>(output_count));
    for (int32_t i = 0; i < output_count; ++i) {
        if (!outputs[i]) return kInvalidPtr;
        geOutputs.push_back(*ToGeOp(outputs[i]));
    }
    graph->graph.SetOutputs(geOutputs);
    return kSuccess;
}

/* ── Query ─────────────────────────────────────────────────────────────── */

CannOperatorHandle cann_graph_find_op_by_name(CannGraphHandle graph, const char* name) {
    if (!graph || !name) return nullptr;
    ge::Operator op = graph->graph.FindOpByName(std::string(name));
    return FromGeOp(new ge::Operator(op));
}

int32_t cann_graph_is_valid(CannGraphHandle graph) {
    if (!graph) return 0;
    return graph->graph.IsValid() ? 1 : 0;
}

}  // extern "C"
}  // namespace ddk
