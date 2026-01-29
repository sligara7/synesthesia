# Market Recorder - Deployment Guide

This guide covers deploying the Market Recorder service to Digital Ocean.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Digital Ocean Droplet                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Market Recorder                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │  Binance    │  │   Signal    │  │     FastAPI     │   │  │
│  │  │  WebSocket  │──│  Detector   │──│       API       │   │  │
│  │  │  Clients    │  │             │  │                 │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │  │
│  │         │               │                  │              │  │
│  │         └───────────────┼──────────────────┘              │  │
│  │                         │                                 │  │
│  │                         ▼                                 │  │
│  │              ┌─────────────────────┐                      │  │
│  │              │    TimescaleDB      │                      │  │
│  │              │  (or DO Managed DB) │                      │  │
│  │              └─────────────────────┘                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Digital Ocean account
- Docker and Docker Compose installed
- Domain (optional, for HTTPS)

## Option 1: Digital Ocean Droplet with Docker

### 1. Create Droplet

Create a new droplet with:
- **Image**: Ubuntu 22.04 LTS
- **Size**: Basic, 2GB RAM / 1 vCPU minimum (4GB recommended for production)
- **Region**: Choose closest to your users
- **Authentication**: SSH key (recommended)

### 2. Initial Server Setup

```bash
# SSH into your droplet
ssh root@your-droplet-ip

# Update packages
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose-plugin -y

# Create app user
useradd -m -s /bin/bash appuser
usermod -aG docker appuser
```

### 3. Deploy Application

```bash
# Switch to app user
su - appuser

# Clone repository
git clone https://github.com/your-repo/synesthesia.git
cd synesthesia/services/market_recorder

# Configure environment
cp .env.example .env
nano .env  # Edit with your settings

# Start services
docker compose up -d

# Check status
docker compose ps
docker compose logs -f market-recorder
```

### 4. Run Database Migrations

```bash
# Connect to TimescaleDB container
docker compose exec timescaledb psql -U synesthesia -d market_data

# Run migrations manually (if not auto-run)
\i /docker-entrypoint-initdb.d/001_initial.sql
\i /docker-entrypoint-initdb.d/002_penetration_signals.sql
\q
```

### 5. Configure Firewall

```bash
# Allow SSH and HTTP
ufw allow 22
ufw allow 8000
ufw enable
```

## Option 2: Digital Ocean Managed Database

For production, use Digital Ocean's Managed PostgreSQL with TimescaleDB:

### 1. Create Managed Database

1. Go to Digital Ocean → Databases → Create Database Cluster
2. Select PostgreSQL 15
3. Choose plan (Basic $15/mo minimum)
4. Select region

### 2. Enable TimescaleDB Extension

```bash
# Connect to managed database
psql "postgresql://user:password@host:25060/defaultdb?sslmode=require"

# Create database and enable TimescaleDB
CREATE DATABASE market_data;
\c market_data
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

### 3. Run Migrations

```bash
# Connect to market_data database
psql "postgresql://user:password@host:25060/market_data?sslmode=require"

# Run migration files
\i migrations/001_initial.sql
\i migrations/002_penetration_signals.sql
```

### 4. Update Environment

```bash
# In your .env file
MARKET_RECORDER_DATABASE_URL=postgresql+asyncpg://user:password@host:25060/market_data?sslmode=require
```

### 5. Deploy App Only

```bash
docker compose -f docker-compose.prod.yml up -d
```

## Option 3: Digital Ocean App Platform

For fully managed deployment:

### 1. Create App

1. Go to Digital Ocean → Apps → Create App
2. Connect your GitHub repository
3. Select branch and source directory: `services/market_recorder`

### 2. Configure Build

```yaml
# app.yaml
name: market-recorder
services:
  - name: api
    dockerfile_path: services/market_recorder/Dockerfile
    source_dir: /
    http_port: 8000
    health_check:
      http_path: /health
    envs:
      - key: MARKET_RECORDER_DATABASE_URL
        scope: RUN_TIME
        value: ${db.DATABASE_URL}
      - key: MARKET_RECORDER_DEFAULT_SYMBOLS
        scope: RUN_TIME
        value: BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT

databases:
  - name: db
    engine: PG
    version: "15"
```

### 3. Add TimescaleDB

Note: App Platform doesn't support TimescaleDB extension directly.
Use a separate Managed Database cluster with TimescaleDB enabled.

## API Usage

### Start Streaming

```bash
# Start streaming top 5 cryptos
curl -X POST http://your-server:8000/streaming/start \
  -H "Content-Type: application/json" \
  -d '{"session_name": "Production Stream"}'
```

### Check Status

```bash
curl http://your-server:8000/streaming/status
```

### Subscribe to Signals (SSE)

```bash
curl -N http://your-server:8000/streaming/signals/stream
```

### Query Historical Signals

```bash
# Get recent signals
curl "http://your-server:8000/streaming/signals?limit=50&min_confidence=0.6"

# Get signals for specific symbol
curl "http://your-server:8000/streaming/signals?symbol=BTCUSDT&signal_type=strong_bullish"
```

### Stop Streaming

```bash
curl -X POST http://your-server:8000/streaming/stop
```

## Monitoring

### Health Check

```bash
curl http://your-server:8000/health
```

### View Logs

```bash
# Docker Compose
docker compose logs -f market-recorder

# Filter for signals
docker compose logs market-recorder | grep -i signal
```

### Database Size

```sql
-- Connect to TimescaleDB
SELECT hypertable_name,
       pg_size_pretty(hypertable_size(format('%I.%I', hypertable_schema, hypertable_name)::regclass))
FROM timescaledb_information.hypertables;
```

## Troubleshooting

### WebSocket Connection Issues

```bash
# Check if Binance WebSocket is accessible
curl -I wss://stream.binance.us:9443/ws

# Check container logs for connection errors
docker compose logs market-recorder | grep -i websocket
```

### Database Connection Issues

```bash
# Test database connection
docker compose exec market-recorder python -c "
from services.market_recorder.database import create_engine
import asyncio
async def test():
    engine = create_engine()
    async with engine.connect() as conn:
        result = await conn.execute('SELECT 1')
        print('DB OK')
asyncio.run(test())
"
```

### High Memory Usage

```bash
# Check container resources
docker stats market-recorder

# Reduce batch size in .env
MARKET_RECORDER_BATCH_INSERT_SIZE=50
```

## Backup & Recovery

### Database Backup

```bash
# Backup to file
docker compose exec timescaledb pg_dump -U synesthesia market_data > backup.sql

# Or use Digital Ocean automated backups for managed database
```

### Restore

```bash
# Restore from backup
docker compose exec -T timescaledb psql -U synesthesia market_data < backup.sql
```

## Scaling

### Vertical Scaling

Upgrade droplet size for:
- More concurrent WebSocket connections
- Faster signal processing
- Larger database cache

### Horizontal Scaling (Future)

For high-frequency requirements:
1. Separate signal detection into its own service
2. Use Redis for pub/sub between services
3. Multiple API instances behind load balancer

## Security Checklist

- [ ] Change default database password
- [ ] Enable firewall (ufw)
- [ ] Use SSH keys, not passwords
- [ ] Configure CORS for your frontend domain
- [ ] Use HTTPS (Let's Encrypt + nginx)
- [ ] Regular security updates
- [ ] Database connection over SSL
