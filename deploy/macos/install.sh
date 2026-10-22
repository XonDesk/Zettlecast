#!/bin/bash
# Run Zettlecast on the Mac mini as the primary (ROLE=primary): the API and the web UI start at boot,
# signed in or not, and restart if they stop. Run from the repository, after setting up .env and
# frontend/.env.local (README.md next to this script):
#
#   ./deploy/macos/install.sh            install / update and (re)start both services
#   ./deploy/macos/install.sh uninstall  stop and remove them
#
# Logs: ~/Library/Logs/Zettlecast/{api,web}.log

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
USER_NAME="$(id -un)"
LOGS="$HOME/Library/Logs/Zettlecast"
DAEMONS=/Library/LaunchDaemons
API=com.zettlecast.api
WEB=com.zettlecast.web
NODE_DIR="$(dirname "$(command -v node || echo /opt/homebrew/bin/node)")"

unload() {
    for label in "$API" "$WEB"; do
        sudo launchctl bootout "system/$label" 2>/dev/null || true
    done
}

if [[ "${1:-}" == "uninstall" ]]; then
    unload
    sudo rm -f "$DAEMONS/$API.plist" "$DAEMONS/$WEB.plist"
    echo "Removed $API and $WEB"
    exit 0
fi

[[ -x "$REPO/.venv/bin/python" ]] || { echo "No $REPO/.venv: run  uv venv && uv pip install -e .  first"; exit 1; }
grep -q '^ROLE=primary' "$REPO/.env" 2>/dev/null || { echo "Set ROLE=primary in $REPO/.env first"; exit 1; }
[[ -d "$REPO/frontend/node_modules" ]] || { echo "Run  npm install  in $REPO/frontend first"; exit 1; }

echo "Building the web UI..."
(cd "$REPO/frontend" && npm run build >/dev/null)
mkdir -p "$LOGS"

plist() {  # label, working dir, log name, program arguments...
    local label=$1 dir=$2 log=$3
    shift 3
    local args=""
    for a in "$@"; do args+="        <string>$a</string>"$'\n'; done
    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>UserName</key><string>$USER_NAME</string>
    <key>WorkingDirectory</key><string>$dir</string>
    <key>ProgramArguments</key>
    <array>
$args    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>$NODE_DIR:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key><string>$HOME</string>
        <key>PYTORCH_ENABLE_MPS_FALLBACK</key><string>1</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>10</integer>
    <key>StandardOutPath</key><string>$LOGS/$log.log</string>
    <key>StandardErrorPath</key><string>$LOGS/$log.log</string>
</dict>
</plist>
EOF
}

unload
# the API answers the worker on the network (its /worker routes need WORKER_TOKEN, the rest API_TOKEN)
plist "$API" "$REPO" api "$REPO/.venv/bin/python" -m uvicorn zettlecast.main:app --host 0.0.0.0 --port 8000 \
    | sudo tee "$DAEMONS/$API.plist" >/dev/null
# the web UI for phones and laptops on the network / VPN (sign-in required)
plist "$WEB" "$REPO/frontend" web "$NODE_DIR/npm" run start -- -H 0.0.0.0 \
    | sudo tee "$DAEMONS/$WEB.plist" >/dev/null
for label in "$API" "$WEB"; do
    sudo chown root:wheel "$DAEMONS/$label.plist"
    sudo chmod 644 "$DAEMONS/$label.plist"
    sudo launchctl bootstrap system "$DAEMONS/$label.plist"
done

echo "Started. API http://$(ipconfig getifaddr en0 2>/dev/null || echo localhost):8000/health ," \
     "web UI :3000. Logs in $LOGS"
