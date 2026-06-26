#!/usr/bin/env bash
# Copy WPT conformance report artifacts into a MkDocs site tree at /wpt-conformance/.
set -euo pipefail

SITE_DIR="${1:-site}"
HTML_SRC="${WPT_HTML:-reports/wpt-conformance.html}"
JSON_SRC="${WPT_JSON:-reports/wpt-conformance.json}"
DEST_DIR="$SITE_DIR/wpt-conformance"

mkdir -p "$DEST_DIR"

if [ -f "$HTML_SRC" ]; then
  cp "$HTML_SRC" "$DEST_DIR/index.html"
  echo "Embedded WPT conformance HTML at $DEST_DIR/index.html"
else
  cat >"$DEST_DIR/index.html" <<'EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>RustNN WPT Conformance</title>
    <style>
      body { font-family: system-ui, sans-serif; max-width: 720px; margin: 48px auto; padding: 0 16px; color: #102a43; }
      h1 { font-size: 1.5rem; }
      p { line-height: 1.5; }
    </style>
  </head>
  <body>
    <h1>RustNN WPT Conformance</h1>
    <p>No conformance report is available yet. The nightly workflow publishes an updated dashboard here after running the in-repo WPT harness.</p>
  </body>
</html>
EOF
  echo "Wrote placeholder WPT conformance page at $DEST_DIR/index.html"
fi

if [ -f "$JSON_SRC" ]; then
  cp "$JSON_SRC" "$DEST_DIR/conformance.json"
  echo "Embedded WPT conformance JSON at $DEST_DIR/conformance.json"
fi
