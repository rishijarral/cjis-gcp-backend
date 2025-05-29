# CJIS Policy Bot - Backend Service

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Technologies Used](#technologies-used)
4.  [Project Structure](#project-structure)
5.  [Prerequisites](#prerequisites)
6.  [Configuration](#configuration)
7.  [Setup and Running](#setup-and-running)
    *   [Using Docker Compose (Recommended)](#using-docker-compose-recommended)
    *   [Manual Setup (Alternative)](#manual-setup-alternative)
8.  [API Endpoints](#api-endpoints)
9.  [Data Management](#data-management)
    *   [Chat Sessions](#chat-sessions)
    *   [Source Documents (for RAG)](#source-documents-for-rag)
10. [Logging](#logging)
11. [SSL and Reverse Proxy](#ssl-and-reverse-proxy)
12. [Scripts](#scripts)

## Overview

This backend service powers the CJIS Policy Bot, providing a RAG (Retrieval Augmented Generation) chat API. It leverages Google Vertex AI's Generative Models and RAG Engine to answer questions based on provided CJIS policy documents. The service includes user authentication, session management, and streams responses for a real-time chat experience.

The system is designed to be containerized using Docker and managed with Docker Compose, incorporating Nginx as a reverse proxy for SSL termination and potentially load balancing.

## Features

*   **RAG Chat API:** Answers questions based on the content of CJIS policy documents.
*   **Vertex AI Integration:** Utilizes Google Vertex AI for LLM and RAG capabilities.
*   **Streaming Responses:** Provides real-time, streamed responses for chat applications.
*   **User Authentication:** Secures endpoints using JWT (Supabase for token validation).
*   **Session Management:** Stores and retrieves chat history for individual users.
*   **Rate Limiting:** Protects the API from abuse.
*   **Health Check Endpoint:** Monitors the service status.
*   **Dockerized:** Easy to deploy and manage using Docker containers.
*   **Nginx Reverse Proxy:** Handles incoming HTTPS traffic and forwards to the API service.
*   **Let's Encrypt SSL:** Automated SSL certificate provisioning and renewal.

## Technologies Used

*   **Backend Framework:** Python 3, FastAPI
*   **AI/ML:** Google Vertex AI (Gemini Models, RAG Engine)
*   **Authentication:** JWT (validated against Supabase)
*   **Containerization:** Docker, Docker Compose
*   **Reverse Proxy:** Nginx
*   **SSL:** Let's Encrypt (via Certbot)
*   **Dynamic DNS:** DuckDNS (for updating IP associated with a domain)
*   **Data Storage:** Filesystem (for chat logs, PDF documents)
*   **Environment Management:** `python-dotenv`

## Project Structure

.
├── Dockerfile # Dockerfile for the FastAPI application
├── Readme.md # This file
├── api
│  └── main.py # Main FastAPI application code
├── cjis-policy-bot-****.json # Google Cloud Service Account Key (renamed for security)
├── data
│  ├── chat_sessions # Stores user chat session logs
│  └── pdf # Contains source PDF documents (e.g., CJIS.pdf) for RAG
├── docker-compose.yml # Defines services, networks, and volumes for Docker
├── duckdns
│  └── duck.sh # Script to update DuckDNS records
├── entrypoint.sh # Entrypoint script for the Docker container
├── letsencrypt # Let's Encrypt SSL certificate data and configuration
├── logs
│  └── container_logs # Stores container logs (if configured)
├── nginx
│  ├── conf.d # Nginx configuration files for sites
│  └── nginx.conf # Main Nginx configuration
├── parsing_script.py # Python script (likely for PDF parsing/processing for RAG)
└── requirements.txt # Python dependencies


## Prerequisites

*   Docker Engine
*   Docker Compose
*   A Google Cloud Platform (GCP) project with Vertex AI API enabled.
*   A GCP Service Account key (`cjis-policy-bot-****.json`) with necessary permissions for Vertex AI.
*   Supabase project for JWT authentication (or compatible JWT issuer).
*   A domain name (if using Let's Encrypt and DuckDNS).

## Configuration

The application relies heavily on environment variables. These should be defined in a `.env` file in the project root or directly in the `docker-compose.yml` file for the `cjis_rag_api_service`.

**Create a `.env` file in the project root with the following (example values):**

```env
# Google Cloud Settings
GOOGLE_APPLICATION_CREDENTIALS=/app/cjis-policy-bot-your-key-file.json # Path inside the container
VERTEX_PROJECT_ID=your-gcp-project-id
VERTEX_LOCATION=us-central1 # Or your GCP region
VERTEX_GEMINI_MODEL_NAME=gemini-2.0-flash-001 # Or desired model Finetuned or not

# RAG Engine Configuration (Ensure RAG Corpus is created in Vertex AI)
RAG_CORPUS_NAME=projects/your-gcp-project-id/locations/your-gcp-region/ragCorpora/your-rag-corpus-id
RAG_TOP_K=3

# API and LLM Settings
# RAG_FT_SYSTEM_INSTRUCTION_FOR_API="Your custom system instruction for the LLM"
GEMINI_SAFETY_SETTINGS_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE # Or BLOCK_NONE, etc.
STREAMING_RESPONSE_TIMEOUT_SECONDS=500
GENERATIVE_MODEL_API_TIMEOUT_SECONDS=90
DISABLE_SAFETY_SETTINGS_FOR_FINETUNED_MODEL=False # Set to True if using a fine-tuned model and want to disable default safety settings

# Supabase JWT Authentication
SUPABASE_URL=https://your-supabase-id.supabase.co
SUPABASE_JWT_SECRET=your-supabase-jwt-secret # If using HS256
# SUPABASE_JWT_AUD=authenticated # Usually 'authenticated' for Supabase, or as configured

# API Server Settings
API_HOST=0.0.0.0
API_PORT=8000
# API_RELOAD_DEV=False # Set to True for development with auto-reload
DEBUG_MODE=False # Set to True for more verbose logging and debug info in responses

# Chat Session Storage
CHAT_SESSIONS_BASE_PATH=/app/data/chat_sessions # Path inside the container

# Rate Limiting
RATE_LIMIT_REQUESTS=15
RATE_LIMIT_WINDOW=60

# Logging
INTERACTION_LOG_PATH=/app/logs/interactions.log # Path inside the container for interaction logs

# Nginx and Let's Encrypt (used by docker-compose.yml and Nginx configs)
DOMAIN_NAME=your.cjis.backend.api.bytecafeanalytics.com # Your domain
LETSENCRYPT_EMAIL=your-email@example.com
DUCKDNS_TOKEN=your-duckdns-token # If using DuckDNS
```

## Important:

    Ensure the GOOGLE_APPLICATION_CREDENTIALS file (e.g., cjis-policy-bot-8965a1c69e2e.json) is present in the project root and its name matches the one in your .env file (or docker-compose.yml).
    Update paths and IDs for RAG_CORPUS_NAME.

## Setup and Running
### Using Docker Compose 

This is the preferred method as it handles the API service, Nginx, and Certbot for SSL.

    Clone the repository (if you haven't already).
    Create and populate your .env file as described in the Configuration section.
    Place your Google Cloud Service Account JSON key file in the project root. Rename it if necessary and update GOOGLE_APPLICATION_CREDENTIALS in your .env or docker-compose.yml.
    Place your PDF document(s) (e.g., CJIS.pdf) into the ./data/pdf/ directory.
    Update Nginx Configuration:
        Modify nginx/conf.d/default.conf to reflect your DOMAIN_NAME.
        If using DuckDNS, ensure duckdns/duck.sh has the correct DUCKDNS_TOKEN and DOMAIN_NAME (these are typically passed as environment variables from docker-compose.yml).
    Build and start the services:

    docker-compose up --build -d

    Initial Let's Encrypt Certificate Setup:
    The docker-compose.yml is set up to automatically request certificates. Monitor the logs of the certbot service:

    docker-compose logs -f certbot

    You might need to run an initial certonly command if there are issues, or adjust Nginx to temporarily serve HTTP for the challenge. Often, Certbot in docker-compose is paired with an entrypoint script that handles this.

### Manual Setup (Alternative)

This is more complex and involves running the FastAPI application directly.

    Install Python 3.10+ and pip.
    Create a virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install dependencies:

    pip install -r requirements.txt

    Set environment variables as described in the Configuration section (e.g., by exporting them in your shell or using a .env file loaded by your script if not using the provided Docker setup).
    Ensure GOOGLE_APPLICATION_CREDENTIALS points to your service account key file.
    Run the FastAPI application:
    The entrypoint.sh script uses Uvicorn. You can run it similarly:

    uvicorn api.main:app --host $API_HOST --port $API_PORT --log-level info

    (Replace $API_HOST and $API_PORT with actual values or ensure they are set as environment variables).

API Endpoints

The API is served by FastAPI, which automatically generates interactive API documentation.

    Swagger UI (Interactive Docs): http://<your_domain_or_ip>:<port>/api/docs
    ReDoc: http://<your_domain_or_ip>:<port>/api/redoc
    OpenAPI Spec: http://<your_domain_or_ip>:<port>/api/openapi.json

If running behind Nginx with SSL, access these via https://<your_domain_name>/api/docs, etc.

Key Endpoints:

    POST /chat: Main endpoint for sending questions and receiving answers. Supports streaming.
    GET /health: Health check for the service.
    GET /api/sessions/: Lists chat sessions for the authenticated user.
    GET /api/sessions/{session_id}: Retrieves messages for a specific chat session.

Data Management
Chat Sessions

    Chat sessions are stored as JSONL files on the filesystem.
    Location: data/chat_sessions/<user_id>/<session_id>.jsonl (as configured by CHAT_SESSIONS_BASE_PATH).
    Each line in the JSONL file represents a user message or a model response.

Source Documents (for RAG)

    PDF documents used by the RAG Engine should be placed in the data/pdf/ directory.
    The RAG_CORPUS_NAME environment variable points to a corpus created in Google Vertex AI, which should be populated with data extracted from these PDFs. The parsing_script.py might be involved in this extraction and upload process, or it's done manually/via another process.

Logging

    Application Logs: The FastAPI application logs to standard output, which is captured by Docker. You can view these logs using:

    docker-compose logs -f cjis_rag_api_service

    Interaction Logs: Detailed request/response logs for the /chat endpoint are stored in the file specified by INTERACTION_LOG_PATH (e.g., /app/logs/interactions.log inside the container).
    Nginx Logs: Also captured by Docker.

    docker-compose logs -f nginx

    Certbot Logs:

    docker-compose logs -f certbot

    Persistent Logs: The logs/container_logs directory in your project structure seems intended for persistent log storage if mounted as a volume, though it's not explicitly shown as mounted for the API service in typical simple setups. The INTERACTION_LOG_PATH within the API service's container can be mapped to this host directory using volumes.

SSL and Reverse Proxy

    Nginx: Handles incoming HTTPS requests, terminates SSL, and forwards traffic to the FastAPI application. Configuration is in nginx/.
    Let's Encrypt: Used for obtaining and renewing SSL certificates. The certbot service in docker-compose.yml manages this. Certificate data is stored in the letsencrypt volume.
    DuckDNS: The duckdns service (running duck.sh) periodically updates your DuckDNS domain with your server's current public IP address. This is crucial if your server is on a dynamic IP.

Scripts

    api/main.py: The main FastAPI application.
    entrypoint.sh: Script executed when the cjis_rag_api_service Docker container starts. It typically waits for other services (if any) and then starts the Uvicorn server.
    duckdns/duck.sh: Script for updating DuckDNS.
    parsing_script.py: Its exact function isn't detailed, but given the context of a RAG system and a CJIS.pdf, it is highly likely used for:
        Extracting text from PDF files (like CJIS.pdf).
        Chunking the extracted text.
        Potentially generating embeddings.
        Uploading the processed data to the Google Vertex AI RAG Corpus specified by RAG_CORPUS_NAME.
        This script would need to be run (possibly manually or as part of a setup job) to populate the RAG Engine with your document content.
