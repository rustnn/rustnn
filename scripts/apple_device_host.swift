// App host for rustnn examples on iOS and watchOS.
//
// An example built as a static library exports `rustnn_example_main`. This host owns the
// application loop, which the system requires of every app, shows one line of text and runs
// the example on a thread with a large stack. `scripts/apple_device_app.py` compiles it.
import Foundation
import SwiftUI

@_silgen_name("rustnn_example_main")
func rustnn_example_main()

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
        }
    }
}
