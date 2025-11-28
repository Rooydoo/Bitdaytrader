# GMO Trading Bot Deployment

This directory contains deployment configurations for running the trading bot as a system service.

## Quick Start

```bash
# Install on VPS
sudo ./install.sh

# Start services
sudo systemctl start gmo-trader
sudo systemctl start gmo-trader-api

# Check status
sudo systemctl status gmo-trader gmo-trader-api
```

## Service Files

### gmo-trader.service
Main trading bot service that:
- Runs the trading engine
- Automatically restarts on crash (after 30 seconds)
- Limits restarts to 5 within 5 minutes
- Memory limit: 2GB
- CPU limit: 50%

### gmo-trader-api.service
Dashboard API server that:
- Runs the FastAPI web server on port 8088
- Automatically restarts on crash (after 10 seconds)
- Memory limit: 512MB
- CPU limit: 25%

## Common Commands

```bash
# Start/Stop/Restart
sudo systemctl start gmo-trader
sudo systemctl stop gmo-trader
sudo systemctl restart gmo-trader

# View logs
sudo journalctl -u gmo-trader -f
sudo journalctl -u gmo-trader-api -f

# View combined logs
sudo tail -f /var/log/gmo-trader/*.log

# Check health
curl http://localhost:8088/api/health
curl http://localhost:8088/api/health/simple
```

## Auto-Restart Behavior

The services are configured with the following restart policy:

- **Trading Bot**: Restarts after 30 seconds delay, max 5 restarts in 5 minutes
- **API Server**: Restarts after 10 seconds delay, max 10 restarts in 5 minutes

If the service fails too many times, systemd will stop trying to restart it.
To reset the failure counter:

```bash
sudo systemctl reset-failed gmo-trader
sudo systemctl start gmo-trader
```

## Monitoring

### Health Check Endpoints

- `GET /api/health` - Detailed health status with all checks
- `GET /api/health/simple` - Simple OK/error for load balancers

### Setting up external monitoring

You can use cron or external monitoring tools to check the health endpoint:

```bash
# Add to crontab for basic monitoring
*/5 * * * * curl -s http://localhost:8088/api/health/simple | grep -q '"status": "ok"' || systemctl restart gmo-trader-api
```

## Security Notes

- Services run under a dedicated `trader` user
- `.env` file permissions are restricted to owner only (600)
- Services use security hardening options (NoNewPrivileges, ProtectSystem, etc.)
- API listens on 0.0.0.0:8088 - use a firewall to restrict access in production
