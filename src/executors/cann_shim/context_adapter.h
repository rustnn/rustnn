/**
 * CANN Adapter Layer - AiContext Wrapper
 *
 * Wraps hiai::AiContext with a pure C interface.
 */

#ifndef CANN_CONTEXT_ADAPTER_H
#define CANN_CONTEXT_ADAPTER_H

#include "adapter_types.h"

namespace ddk {
extern "C" {

/* ── Context lifecycle ───────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannContextHandle cann_context_create();
CANN_ADAPTER_EXPORT void              cann_context_destroy(CannContextHandle context);

/* ── Key-Value parameters ────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_context_set_para(CannContextHandle context,
                                   const char* key,
                                   const char* value);

CANN_ADAPTER_EXPORT const char* cann_context_get_para(CannContextHandle context,
                                    const char* key);

CANN_ADAPTER_EXPORT CannStatus cann_context_add_para(CannContextHandle context,
                                   const char* key,
                                   const char* value);

CANN_ADAPTER_EXPORT CannStatus cann_context_del_para(CannContextHandle context,
                                   const char* key);

CANN_ADAPTER_EXPORT CannStatus cann_context_clear_para(CannContextHandle context);

CANN_ADAPTER_EXPORT CannStatus cann_context_get_all_keys(CannContextHandle context,
                                       char** keys,
                                       int32_t max_keys,
                                       int32_t* out_key_count);

}  // extern "C"
}  // namespace ddk

#endif /* CANN_CONTEXT_ADAPTER_H */
