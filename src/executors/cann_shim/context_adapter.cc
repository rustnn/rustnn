/**
 * CANN Adapter Layer - AiContext Implementation
 */

#include "context_adapter.h"
#include "adapter_internal.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace ddk {
extern "C" {

/* ── Context lifecycle ──────────────────────────────────────────────────── */

CannContextHandle cann_context_create() {
    auto* ctx = new CannContextImpl();
    return reinterpret_cast<CannContextHandle>(ctx);
}

void cann_context_destroy(CannContextHandle context) {
    delete reinterpret_cast<CannContextImpl*>(context);
}

/* ── Key-Value parameters ──────────────────────────────────────────────── */

CannStatus cann_context_set_para(CannContextHandle context,
                                   const char* key,
                                   const char* value) {
    if (!context || !key || !value) return kInvalidPtr;
    reinterpret_cast<CannContextImpl*>(context)->context.SetPara(
        std::string(key), std::string(value));
    return kSuccess;
}

const char* cann_context_get_para(CannContextHandle context,
                                    const char* key) {
    if (!context || !key) return nullptr;
    std::string val =
        reinterpret_cast<CannContextImpl*>(context)->context.GetPara(std::string(key));
    if (val.empty()) return nullptr;
    char* buf = static_cast<char*>(std::malloc(val.size() + 1));
    if (!buf) return nullptr;
    std::memcpy(buf, val.c_str(), val.size() + 1);
    return buf; /* Caller must free() */
}

CannStatus cann_context_add_para(CannContextHandle context,
                                   const char* key,
                                   const char* value) {
    if (!context || !key || !value) return kInvalidPtr;
    reinterpret_cast<CannContextImpl*>(context)->context.AddPara(
        std::string(key), std::string(value));
    return kSuccess;
}

CannStatus cann_context_del_para(CannContextHandle context,
                                   const char* key) {
    if (!context || !key) return kInvalidPtr;
    reinterpret_cast<CannContextImpl*>(context)->context.DelPara(std::string(key));
    return kSuccess;
}

CannStatus cann_context_clear_para(CannContextHandle context) {
    if (!context) return kInvalidPtr;
    reinterpret_cast<CannContextImpl*>(context)->context.ClearPara();
    return kSuccess;
}

CannStatus cann_context_get_all_keys(CannContextHandle context,
                                       char** keys,
                                       int32_t max_keys,
                                       int32_t* out_key_count) {
    if (!context || !keys || max_keys <= 0 || !out_key_count)
        return kInvalidPara;
    std::vector<std::string> keyVec;
    reinterpret_cast<CannContextImpl*>(context)->context.GetAllKeys(keyVec);
    *out_key_count = static_cast<int32_t>(keyVec.size());
    for (size_t i = 0; i < keyVec.size() && static_cast<int32_t>(i) < max_keys; ++i) {
        char* buf = static_cast<char*>(std::malloc(keyVec[i].size() + 1));
        if (buf) {
            std::memcpy(buf, keyVec[i].c_str(), keyVec[i].size() + 1);
            keys[i] = buf;
        }
    }
    return kSuccess;
}

}  // extern "C"
}  // namespace ddk
