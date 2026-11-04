#!/bin/bash
# The native Wake-on-LAN helper for an API running in a container (WAKE_SENDER=helper): launchd runs
# wake-helper.py whenever <storage>/wake_requests/ has a request in it. A LaunchAgent: no sudo.
#
#   ./deploy/macos/install-wake-helper.sh             install / update
#   ./deploy/macos/install-wake-helper.sh uninstall   remove
#
# Log: ~/Library/Logs/Zettlecast/wake.log

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL=com.zettlecast.wake
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGS="$HOME/Library/Logs/Zettlecast"
STORAGE="$(grep -E '^STORAGE_PATH=' "$REPO/.env" 2>/dev/null | cut -d= -f2- | sed 's/ #.*//' || true)"
STORAGE="${STORAGE:-$HOME/_BRAIN_STORAGE}"
QUEUE="$STORAGE/wake_requests"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [[ "${1:-}" == "uninstall" ]]; then
    rm -f "$PLIST"
    echo "Removed $LABEL"
    exit 0
fi

mkdir -p "$QUEUE" "$LOGS" "$(dirname "$PLIST")"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$REPO/deploy/macos/wake-helper.py</string>
        <string>$QUEUE</string>
    </array>
    <key>QueueDirectories</key>
    <array><string>$QUEUE</string></array>
    <key>ThrottleInterval</key><integer>5</integer>
    <key>StandardOutPath</key><string>$LOGS/wake.log</string>
    <key>StandardErrorPath</key><string>$LOGS/wake.log</string>
</dict>
</plist>
EOF
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed $LABEL: sends the wake requests written to $QUEUE (log: $LOGS/wake.log)"
