#!/bin/bash
# Build a real macOS "JARVIS.app" you can keep in Applications or the Dock.
# Double-clicking the app opens Terminal and runs the JARVIS voice console.
#
#   bash scripts/make_app.sh            # builds to ~/Applications/JARVIS.app
#   bash scripts/make_app.sh ~/Desktop  # or into any folder you pass
#
# Re-run any time; it overwrites the existing app. The app just points back at
# this repo's launcher, so updating the code needs no rebuild.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$SCRIPT_DIR/Start JARVIS.command"

DEST_DIR="${1:-$HOME/Applications}"
APP="$DEST_DIR/JARVIS.app"

mkdir -p "$DEST_DIR"
chmod +x "$LAUNCHER"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# Info.plist — names the app and its executable.
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>JARVIS</string>
  <key>CFBundleDisplayName</key><string>JARVIS</string>
  <key>CFBundleIdentifier</key><string>com.akshath.jarvis</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>JARVIS</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# The bundle executable: open Terminal and run our launcher. Using Terminal (not
# a background exec) so the user sees logs and can Ctrl-C to stop.
cat > "$APP/Contents/MacOS/JARVIS" <<LAUNCH
#!/bin/bash
open -a Terminal "$LAUNCHER"
LAUNCH
chmod +x "$APP/Contents/MacOS/JARVIS"

# Give it JARVIS's HUD favicon as the app icon if one exists (best-effort).
FAVICON="$REPO_ROOT/jarvis/ui/static/favicon.png"
if command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1 && [[ -f "$FAVICON" ]]; then
  TMP_ICONSET="$(mktemp -d)/JARVIS.iconset"
  mkdir -p "$TMP_ICONSET"
  for sz in 16 32 128 256 512; do
    sips -z "$sz" "$sz"       "$FAVICON" --out "$TMP_ICONSET/icon_${sz}x${sz}.png"      >/dev/null 2>&1 || true
    sips -z "$((sz*2))" "$((sz*2))" "$FAVICON" --out "$TMP_ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null 2>&1 || true
  done
  iconutil -c icns "$TMP_ICONSET" -o "$APP/Contents/Resources/JARVIS.icns" >/dev/null 2>&1 \
    && /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string JARVIS" "$APP/Contents/Info.plist" >/dev/null 2>&1 || true
fi

# Refresh Finder/Launch Services so the icon shows immediately.
touch "$APP"
echo "✓  Built $APP"
echo "   Double-click it, or drag it to your Dock / Applications."
