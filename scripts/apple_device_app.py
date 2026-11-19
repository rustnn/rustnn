#!/usr/bin/env python3
"""Package a rustnn example as a signed iOS or watchOS app and run it on a device.

The examples are command-line programs. On a device they need an application loop, an app
bundle, a development signature and a way to hand back their result. This script links an
example built as a static library with a small SwiftUI host (`apple_device_host.swift`),
adds its data directory, signs `<name>.app` with your development identity, installs it
with `devicectl`, launches it with arguments and waits for the receipt file the example
writes to the app's `Library/Caches`.

    apple_device_app.py package --platform ios --library <lib.a> --data <dir> \\
        --bundle-id com.example.smollm2 --team <TEAMID> --identity "Apple Development: ..." \\
        --profile <file.mobileprovision> --out build/ios
    apple_device_app.py run --device <udid> --app build/ios/Example.app \\
        --receipt receipt.json --timeout 900 -- --bundle model --receipt @RECEIPT@

`@RECEIPT@` in the arguments is replaced by a file name that is unique to the launch. The
app's container keeps files across launches and reinstalls, so a fixed name can return the
receipt of an earlier run.

Requirements: Xcode with the platform SDK, a development certificate, and a provisioning
profile that covers the bundle identifier and the device. The device must be unlocked and
awake for the launch. Only the Python standard library is used.
"""
import argparse
import json
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

PLATFORMS = {
    # platform: (CFBundleSupportedPlatforms, UIDeviceFamily, extra Info.plist keys)
    "ios": ("iPhoneOS", [1, 2], {"LSRequiresIPhoneOS": True, "UILaunchScreen": {}}),
    "watchos": ("WatchOS", [4], {"WKApplication": True, "WKWatchOnly": True}),
}


SDKS = {"ios": ("iphoneos", "arm64-apple-ios"), "watchos": ("watchos", "arm64-apple-watchos")}


def package(args):
    supported, family, extra = PLATFORMS[args.platform]
    app = Path(args.out) / f"{args.name}.app"
    if app.exists():
        shutil.rmtree(app)
    app.mkdir(parents=True)
    sdk, triple = SDKS[args.platform]
    host = Path(__file__).with_name("apple_device_host.swift")
    library = Path(args.library).resolve()
    subprocess.run(
        [
            "xcrun", "--sdk", sdk, "swiftc", "-O", "-parse-as-library",
            "-target", triple + args.minimum_os, str(host), str(library),
            "-framework", "CoreML", "-framework", "Foundation",
            "-lc++", "-lobjc", "-liconv",
            "-o", str(app / args.name),
        ],
        check=True,
    )
    if args.data:
        shutil.copytree(args.data, app / args.data_name)
    info = {
        "CFBundleIdentifier": args.bundle_id,
        "CFBundleName": args.name,
        "CFBundleDisplayName": args.name,
        "CFBundleExecutable": args.name,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleSupportedPlatforms": [supported],
        "MinimumOSVersion": args.minimum_os,
        "UIDeviceFamily": family,
        **extra,
    }
    (app / "Info.plist").write_bytes(plistlib.dumps(info))
    shutil.copyfile(args.profile, app / "embedded.mobileprovision")
    entitlements = Path(args.out) / "development.entitlements"
    entitlements.write_bytes(
        plistlib.dumps(
            {
                "application-identifier": f"{args.team}.{args.bundle_id}",
                "com.apple.developer.team-identifier": args.team,
                "get-task-allow": True,
            }
        )
    )
    sign = ["codesign", "--force", "--sign", args.identity, "--entitlements", str(entitlements)]
    subprocess.run(sign + ["--generate-entitlement-der", "--timestamp=none", str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--strict", str(app)], check=True)
    print(app)


def devicectl(*arguments, check=True):
    result = subprocess.run(["xcrun", "devicectl", *arguments], capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(f"devicectl {' '.join(arguments[:3])} failed:\n{result.stdout}{result.stderr}")
    return result


def run(args):
    app = Path(args.app)
    bundle_id = plistlib.loads((app / "Info.plist").read_bytes())["CFBundleIdentifier"]
    remote = f"receipt-{int(time.time())}.json"
    arguments = [remote if a == "@RECEIPT@" else a for a in args.arguments]
    devicectl("device", "install", "app", "--device", args.device, str(app))
    launch = Path(args.receipt).with_suffix(".launch.json")
    devicectl(
        "device", "process", "launch", "--device", args.device, "--terminate-existing",
        "--json-output", str(launch), bundle_id, *arguments,
    )
    process = json.loads(launch.read_text())["result"]["process"]["processIdentifier"]
    receipt = Path(args.receipt)
    receipt.unlink(missing_ok=True)
    deadline = time.time() + args.timeout
    while time.time() < deadline and not (receipt.exists() and receipt.stat().st_size):
        time.sleep(4)
        devicectl(
            "device", "copy", "from", "--device", args.device,
            "--domain-type", "appDataContainer", "--domain-identifier", bundle_id,
            "--source", f"Library/Caches/{remote}", "--destination", str(receipt),
            check=False,
        )
    devicectl("device", "process", "terminate", "--device", args.device, "--pid", str(process), check=False)
    if not (receipt.exists() and receipt.stat().st_size):
        sys.exit(f"no receipt after {args.timeout} s; is the device unlocked and awake?")
    print(receipt.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("package", help="wrap and sign a binary as an app")
    p.add_argument("--platform", choices=PLATFORMS, required=True)
    p.add_argument("--library", required=True, help="example built with --crate-type staticlib")
    p.add_argument("--data", help="directory copied into the app next to the executable")
    p.add_argument("--data-name", default="model", help="name of the data directory in the app")
    p.add_argument("--name", default="Example", help="app and executable name")
    p.add_argument("--bundle-id", required=True)
    p.add_argument("--team", required=True, help="development team identifier")
    p.add_argument("--identity", required=True, help="codesign identity (name or SHA-1)")
    p.add_argument("--profile", required=True, help="provisioning profile file")
    p.add_argument("--minimum-os", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(function=package)
    r = commands.add_parser("run", help="install, launch and fetch the receipt")
    r.add_argument("--device", required=True, help="device identifier from `xcrun devicectl list devices`")
    r.add_argument("--app", required=True)
    r.add_argument("--receipt", required=True, help="local path for the fetched receipt")
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("arguments", nargs=argparse.REMAINDER, help="-- followed by the app's arguments")
    r.set_defaults(function=run)
    args = parser.parse_args()
    if args.command == "package" and args.minimum_os is None:
        args.minimum_os = {"ios": "18.0", "watchos": "11.0"}[args.platform]
    if args.command == "run" and args.arguments[:1] == ["--"]:
        args.arguments = args.arguments[1:]
    args.function(args)


if __name__ == "__main__":
    main()
