#!/bin/bash
#
# GMO Trading Bot Installation Script
#
# Usage: sudo ./install.sh
#
set -e

INSTALL_DIR="/opt/bitdaytrader"
SERVICE_USER="trader"
LOG_DIR="/var/log/gmo-trader"

echo "=== GMO Trading Bot Installer ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: Please run as root (sudo ./install.sh)"
    exit 1
fi

# Create service user if not exists
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "Creating service user: $SERVICE_USER"
    useradd --system --create-home --shell /bin/bash "$SERVICE_USER"
fi

# Create directories
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/models"

# Copy files (assuming running from repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "Copying files from $REPO_DIR to $INSTALL_DIR..."
cp -r "$REPO_DIR/src" "$INSTALL_DIR/"
cp -r "$REPO_DIR/config" "$INSTALL_DIR/"
cp -r "$REPO_DIR/dashboard" "$INSTALL_DIR/"
cp "$REPO_DIR/pyproject.toml" "$INSTALL_DIR/"
cp "$REPO_DIR/requirements.txt" "$INSTALL_DIR/"

# Copy .env if exists
if [ -f "$REPO_DIR/.env" ]; then
    cp "$REPO_DIR/.env" "$INSTALL_DIR/"
    chmod 600 "$INSTALL_DIR/.env"
else
    echo "Warning: .env file not found. Please create it manually."
    cp "$REPO_DIR/.env.example" "$INSTALL_DIR/.env"
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# Set ownership
echo "Setting permissions..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
chmod 755 "$INSTALL_DIR"

# Install systemd services
echo "Installing systemd services..."
cp "$SCRIPT_DIR/gmo-trader.service" /etc/systemd/system/
cp "$SCRIPT_DIR/gmo-trader-api.service" /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable services
echo "Enabling services..."
systemctl enable gmo-trader.service
systemctl enable gmo-trader-api.service

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit configuration: sudo nano $INSTALL_DIR/.env"
echo "2. Start the trading bot: sudo systemctl start gmo-trader"
echo "3. Start the API server: sudo systemctl start gmo-trader-api"
echo "4. Check status: sudo systemctl status gmo-trader gmo-trader-api"
echo "5. View logs: sudo journalctl -u gmo-trader -f"
echo ""
echo "Dashboard URL: http://localhost:8088"
echo "Health check: curl http://localhost:8088/api/health"
