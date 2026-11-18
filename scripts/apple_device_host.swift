// App host for rustnn examples on iOS and watchOS.
//
// An example built as a static library exports `rustnn_example_main`. This host owns the
// application loop, which the system requires of every app, shows one line of text and runs
// the example on a thread with a large stack. `scripts/apple_device_app.py` compiles it.
import Foundation
import SwiftUI
#if os(watchOS)
import WatchKit
#endif

@_silgen_name("rustnn_example_main")
func rustnn_example_main()

#if os(watchOS)
/// watchOS suspends an app soon after the screen sleeps, which would stop the run in the
/// middle of a model load. An extended runtime session keeps the app running for its
/// allowed time (the app's Info.plist declares the `mindfulness` background mode for it).
final class KeepRunning: NSObject, WKExtendedRuntimeSessionDelegate {
    static let shared = KeepRunning()
    private var session: WKExtendedRuntimeSession?

    func start() {
        let session = WKExtendedRuntimeSession()
        session.delegate = self
        session.start()
        self.session = session
    }

    func extendedRuntimeSessionDidStart(_ session: WKExtendedRuntimeSession) {}
    func extendedRuntimeSessionWillExpire(_ session: WKExtendedRuntimeSession) {}
    func extendedRuntimeSession(
        _ session: WKExtendedRuntimeSession,
        didInvalidateWith reason: WKExtendedRuntimeSessionInvalidationReason,
        error: Error?
    ) {
        NSLog("extended runtime session ended: %ld %@", reason.rawValue, String(describing: error))
    }
}
#endif

@main
struct ExampleHost: App {
    init() {
        let thread = Thread { rustnn_example_main() }
        thread.stackSize = 64 << 20
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    var body: some Scene {
        WindowGroup {
            Text("rustnn example running")
                #if os(watchOS)
                .onAppear { KeepRunning.shared.start() }
                #endif
        }
    }
}
