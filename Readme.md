# CJIS Backend API Server

This project hosts the backend infrastructure for the CJIS (Criminal Justice Information Services) API service. It is containerized using Docker and served via Nginx as a reverse proxy, with HTTPS support enabled through Let's Encrypt.

---

##  Server Architecture Overview

The server stack includes:

- **Nginx**: Acts as a reverse proxy to route incoming HTTPS requests to the backend API service.
- **Flask API (Python)**: The core application, located in `api/main.py`, serves REST endpoints.
- **Certbot (Let's Encrypt)**: Automatically manages and renews TLS/SSL certificates for secure HTTPS.
- **DuckDNS**: Used for dynamic DNS updates (useful for hosting on a dynamic IP).

---

##  How Requests Flow

1. **Client** sends an HTTPS request to `https://cjis.backend.api.bytecafeanalytics.com`.
2. **Nginx** receives the request and terminates SSL using Let's Encrypt certificates.
3. **Nginx** forwards the request to the backend container running the Flask app on an internal port (e.g., `http://api:5000`).
4. **The Flask API** handles the request and sends a response back through Nginx to the client.

---

## Key Files & Directories

| Path                       | Purpose                                            |
|----------------------------|----------------------------------------------------|
| `nginx/conf.d/default.conf`| Nginx site config for reverse proxy + SSL          |
| `letsencrypt/`             | Stores certificates and renewal configs            |
| `duckdns/duck.sh`          | Script for updating DuckDNS IP dynamically         |
| `docker-compose.yml`       | Orchestrates all containers                        |
| `entrypoint.sh`            | Custom init script for container setup             |
| `api/main.py`              | Flask application entry point                      |

---

## Security & Best Practices

- TLS certificates are auto-managed by Certbot and renewed via cron.
- Nginx is configured to enforce HTTPS using strong cipher suites and Diffie-Hellman parameters.
- Backend services are only accessible internally and not exposed to the public internet.
- Secrets (like service account JSON) are excluded via `.gitignore`.

---

##  Deployment

To deploy locally or on a VM:

```bash
git clone git@github.com:rishijarral/cjis-gcp-backend.git
cd cjis-gcp-backend
docker-compose up --build -d
