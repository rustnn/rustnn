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
#include <stddef.h>
#include <stdio.h>

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
// borrowed (autoreleased) NSURL valid for the caller's current autorelease pool.
int rustnn_coreml_compile(void *model_url, void **out_url, char *err, size_t err_len) {
    *out_url = NULL;
    @try {
        NSError *nserr = nil;
        NSURL *compiled = [MLModel compileModelAtURL:(__bridge NSURL *)model_url error:&nserr];
        if (compiled == nil) {
            rustnn_copy_err(err, err_len,
                            nserr ? [nserr localizedDescription] : @"MLModel compile failed");
            return 1;
        }
        *out_url = (__bridge void *)compiled;
        return 0;
    } @catch (NSException *e) {
        rustnn_copy_err(err, err_len,
                        [NSString stringWithFormat:@"%@: %@", e.name, e.reason]);
        return 2;
    } @catch (...) {
        rustnn_copy_err(err, err_len, @"caught non-Objective-C exception during CoreML compile");
        return 3;
    }
}

// Load an MLModel from a compiled URL with the given configuration. On success
// `*out_model` is a borrowed (autoreleased) MLModel valid for the caller's
// current autorelease pool; callers that need it to outlive the pool must
// `retain` it themselves (as compile_model does).
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
        *out_model = (__bridge void *)model;
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

// Run a prediction. On success `*out_provider` is a borrowed (autoreleased)
// output feature provider valid for the caller's current autorelease pool.
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
        *out_provider = (__bridge void *)out;
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

} // extern "C"
