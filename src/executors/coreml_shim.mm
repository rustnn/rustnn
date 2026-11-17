// Objective-C++ exception firewall for the CoreML runtime.
//
// A few CoreML calls (model compilation, model loading, and prediction) can
// raise Objective-C `NSException`s or, worse, C++/STL exceptions from deep
// inside Espresso ("E5RT encountered an STL exception ..."). Those exceptions
// propagate out of `objc_msgSend`, which Rust declares as `extern "C"`. A
// foreign exception unwinding across that boundary aborts the whole process
// ("fatal runtime error: Rust cannot catch foreign exceptions"), which would
// take down the entire test suite on a single malformed model.
//
// The only place such an exception can be caught is where `objc_msgSend` is
// actually issued, so these wrappers perform the risky calls here inside
// `@try/@catch` (including `@catch (...)` for C++ exceptions) and hand a plain
// status code plus an error string back to Rust.
//
// Return codes: 0 = success, 1 = call returned nil with an NSError,
// 2 = Objective-C NSException caught, 3 = non-Objective-C (C++) exception caught.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

extern "C" {

static void rustnn_copy_err(char *buf, size_t len, NSString *msg) {
    if (buf == NULL || len == 0) {
        return;
    }
    const char *c = (msg != nil) ? [msg UTF8String] : "unknown CoreML error";
    if (c == NULL) {
        c = "unknown CoreML error";
    }
    // snprintf always NUL-terminates within `len`.
    snprintf(buf, len, "%s", c);
}

// Compile a `.mlmodel`/`.mlpackage` at `model_url`. On success `*out_url` is a
// RETAINED NSURL; the caller owns it and must release it when done.
//
// Same ownership hazard as `rustnn_coreml_predict` below: the +0 return of
// `compileModelAtURL:` may never enter an autorelease pool when ARC's
// return-value hand-off elides the autorelease, so a plain `__bridge` cast can
// hand the caller an already-deallocated object in optimized builds.
int rustnn_coreml_compile(void *model_url, void **out_url, char *err, size_t err_len) {
    *out_url = NULL;
#if TARGET_OS_WATCH
    // watchOS has no public `compileModelAtURL:`; the `coreml-export` route
    // writes the bundle in Rust instead.
    (void)model_url;
    rustnn_copy_err(err, err_len, @"MLModel compileModelAtURL: is unavailable on watchOS");
    return 1;
#else
    @try {
        NSError *nserr = nil;
        NSURL *compiled = [MLModel compileModelAtURL:(__bridge NSURL *)model_url error:&nserr];
        if (compiled == nil) {
            rustnn_copy_err(err, err_len,
                            nserr ? [nserr localizedDescription] : @"MLModel compile failed");
            return 1;
        }
        *out_url = (__bridge_retained void *)compiled;
        return 0;
    } @catch (NSException *e) {
        rustnn_copy_err(err, err_len,
                        [NSString stringWithFormat:@"%@: %@", e.name, e.reason]);
        return 2;
    } @catch (...) {
        rustnn_copy_err(err, err_len, @"caught non-Objective-C exception during CoreML compile");
        return 3;
    }
#endif
}

// Load an MLModel from a compiled URL with the given configuration. On success
// `*out_model` is a RETAINED MLModel; the caller owns it and must release it
// when done (see the ownership note on `rustnn_coreml_compile`).
int rustnn_coreml_load(void *compiled_url, void *config, void **out_model, char *err,
                       size_t err_len) {
    *out_model = NULL;
    @try {
        NSError *nserr = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:(__bridge NSURL *)compiled_url
                                           configuration:(__bridge MLModelConfiguration *)config
                                                   error:&nserr];
        if (model == nil) {
            rustnn_copy_err(err, err_len,
                            nserr ? [nserr localizedDescription] : @"MLModel load failed");
            return 1;
        }
        *out_model = (__bridge_retained void *)model;
        return 0;
    } @catch (NSException *e) {
        rustnn_copy_err(err, err_len,
                        [NSString stringWithFormat:@"%@: %@", e.name, e.reason]);
        return 2;
    } @catch (...) {
        rustnn_copy_err(err, err_len, @"caught non-Objective-C exception during CoreML load");
        return 3;
    }
}

// Run a prediction. On success `*out_provider` is a RETAINED output feature
// provider; the caller owns it and must release it when done.
//
// `-predictionFromFeatures:error:` returns its result at +0 (autoreleased);
// under ARC, `out` (a strong local) picks up a retain for the duration of this
// function and ARC releases it again when `out` goes out of scope at `return`.
// A plain `__bridge` cast here does not add a matching retain of its own, so by
// the time control returns to the caller the object can already be
// deallocated -- reproduced 100% of the time in optimized (non-debug) builds as
// a SIGSEGV (garbage receiver in `objc_msgSend`) on the very next use of
// `*out_provider`. `__bridge_retained` performs the retain the caller needs to
// keep the object alive past this function returning.
int rustnn_coreml_predict(void *model, void *features, void **out_provider, char *err,
                          size_t err_len) {
    *out_provider = NULL;
    @try {
        NSError *nserr = nil;
        id<MLFeatureProvider> out =
            [(__bridge MLModel *)model predictionFromFeatures:(__bridge id<MLFeatureProvider>)features
                                                        error:&nserr];
        if (out == nil) {
            rustnn_copy_err(err, err_len,
                            nserr ? [nserr localizedDescription] : @"prediction returned nil");
            return 1;
        }
        *out_provider = (__bridge_retained void *)out;
        return 0;
    } @catch (NSException *e) {
        rustnn_copy_err(err, err_len,
                        [NSString stringWithFormat:@"%@: %@", e.name, e.reason]);
        return 2;
    } @catch (...) {
        rustnn_copy_err(err, err_len, @"caught non-Objective-C exception during CoreML prediction");
        return 3;
    }
}

// Retained shape view over a context-owned buffer. The context
// keeps the buffer alive longer than the view and every synchronous prediction.
int rustnn_coreml_array_view(void *data, const int64_t *shape, const int64_t *strides,
                            size_t rank, int32_t dtype, void **out, char *err, size_t err_len) {
    *out = NULL;
    @try {
        NSMutableArray<NSNumber *> *dimensions = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber *> *steps = [NSMutableArray arrayWithCapacity:rank];
        for (size_t i = 0; i < rank; ++i) {
            [dimensions addObject:@(shape[i])];
            [steps addObject:@(strides[i])];
        }
        NSError *nserr = nil;
        MLMultiArray *array = [[MLMultiArray alloc] initWithDataPointer:data shape:dimensions
            dataType:(MLMultiArrayDataType)dtype strides:steps deallocator:nil error:&nserr];
        if (array == nil) {
            rustnn_copy_err(err, err_len, nserr.localizedDescription ?: @"array view failed");
            return 1;
        }
        *out = (__bridge_retained void *)array;
        return 0;
    } @catch (NSException *e) {
        rustnn_copy_err(err, err_len, [NSString stringWithFormat:@"%@: %@", e.name, e.reason]);
        return 2;
    } @catch (...) {
        rustnn_copy_err(err, err_len, @"exception creating native tensor view");
        return 3;
    }
}

int rustnn_coreml_predict_backed(void *model, void *features, void *backings,
                                void **out_provider, char *err, size_t err_len) {
    *out_provider = NULL;
    @try {
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        if ([options respondsToSelector:@selector(setOutputBackings:)]) {
            options.outputBackings = (__bridge NSDictionary *)backings;
        }
        NSError *nserr = nil;
        id<MLFeatureProvider> out = [(__bridge MLModel *)model
            predictionFromFeatures:(__bridge id<MLFeatureProvider>)features options:options error:&nserr];
        if (out == nil) {
            rustnn_copy_err(err, err_len, nserr.description ?: @"prediction returned nil");
            return 1;
        }
        *out_provider = (__bridge_retained void *)out;
        return 0;
    } @catch (NSException *e) {
        rustnn_copy_err(err, err_len, [NSString stringWithFormat:@"%@: %@", e.name, e.reason]);
        return 2;
    } @catch (...) {
        rustnn_copy_err(err, err_len, @"exception during backed CoreML prediction");
        return 3;
    }
}

// Create a fresh, zero-initialized MLState for a model whose carried-state pairs live
// inside the compiled program (`read_state` / `write_state`, iOS 18 / macOS 15). On
// success `*out_state` is a RETAINED MLState; the caller owns it (`-newState` returns
// +1 and `__bridge_retained` adds the caller's retain; ARC releases the local).
int rustnn_coreml_new_state(void *model, void **out_state, char *err, size_t err_len) {
    *out_state = NULL;
    @try {
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)) {
            MLState *state = [(__bridge MLModel *)model newState];
            if (state == nil) {
                rustnn_copy_err(err, err_len, @"MLModel newState returned nil");
                return 1;
            }
            *out_state = (__bridge_retained void *)state;
            return 0;
        }
        rustnn_copy_err(err, err_len, @"CoreML model states require iOS 18 or macOS 15");
        return 1;
    } @catch (NSException *e) {
        rustnn_copy_err(err, err_len, [NSString stringWithFormat:@"%@: %@", e.name, e.reason]);
        return 2;
    } @catch (...) {
        rustnn_copy_err(err, err_len, @"exception creating a CoreML model state");
        return 3;
    }
}

// Run a prediction that reads and updates `state` in place. `backings` may be NULL.
// Ownership of `*out_provider` follows `rustnn_coreml_predict`.
int rustnn_coreml_predict_stateful(void *model, void *features, void *state, void *backings,
                                   void **out_provider, char *err, size_t err_len) {
    *out_provider = NULL;
    @try {
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)) {
            MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
            if (backings != NULL && [options respondsToSelector:@selector(setOutputBackings:)]) {
                options.outputBackings = (__bridge NSDictionary *)backings;
            }
            NSError *nserr = nil;
            id<MLFeatureProvider> out = [(__bridge MLModel *)model
                predictionFromFeatures:(__bridge id<MLFeatureProvider>)features
                            usingState:(__bridge MLState *)state
                               options:options
                                 error:&nserr];
            if (out == nil) {
                rustnn_copy_err(err, err_len,
                                nserr.localizedDescription ?: @"stateful prediction returned nil");
                return 1;
            }
            *out_provider = (__bridge_retained void *)out;
            return 0;
        }
        rustnn_copy_err(err, err_len, @"CoreML model states require iOS 18 or macOS 15");
        return 1;
    } @catch (NSException *e) {
        rustnn_copy_err(err, err_len, [NSString stringWithFormat:@"%@: %@", e.name, e.reason]);
        return 2;
    } @catch (...) {
        rustnn_copy_err(err, err_len, @"exception during stateful CoreML prediction");
        return 3;
    }
}

} // extern "C"
