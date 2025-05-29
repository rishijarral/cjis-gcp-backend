

# CJIS Policy Parser and RAG Backend

This backend extracts structured data from the official CJIS policy PDF and prepares it for two tasks:

- Fine-tuning Gemini models
- Retrieval-Augmented Generation (RAG) with FAISS

Optimized for GPU acceleration and cloud portability, it runs locally or on GCP VMs.

## Project Structure

```
.
├── Dockerfile               # CUDA-enabled image with FAISS GPU support
├── docker-compose.yml       # Defines parser and vector DB services
├── parsing_script.py        # Parses PDF and generates JSONL output
├── setup_vector_db.py       # Builds FAISS vector index
├── requirements.txt         # Python dependencies
├── data/
│   ├── pdf/                 # CJIS.pdf goes here
│   └── output_jsonl/        # Fine-tuning and RAG JSONL outputs
├── my_vector_store/         # FAISS index and metadata
└── backup.txt               # Optional scratch or notes
```

## Prerequisites

- Docker
- NVIDIA GPU and drivers
- NVIDIA Container Toolkit
- (Optional) Google Cloud CLI for Vertex AI

## Setup

Build the Docker image:

```bash
docker compose build --no-cache
```

## Step 1: Generate JSONL

Parse the input PDF:

```bash
docker compose up pdf-parser
```

Produces:

- `data/output_jsonl/cjis_finetune_dataset.jsonl`
- `data/output_jsonl/cjis_rag_data.jsonl`

## Step 2: Create Vector Database

Build the FAISS index:

```bash
docker compose run vector-db
```

Produces:

- `my_vector_store/cjis.index`
- `my_vector_store/cjis_docs.pkl`

## Environment Variables

Override file paths via environment variables:

```bash
INPUT_PDF_PATH=./data/pdf/CJIS.pdf
FT_JSONL_PATH=./data/output_jsonl/cjis_finetune_dataset.jsonl
RAG_JSONL_PATH=./data/output_jsonl/cjis_rag_data.jsonl
FAISS_INDEX_PATH=./my_vector_store/cjis.index
METADATA_PATH=./my_vector_store/cjis_docs.pkl
```

## Deployment on GCP VM

- Clone the project to your VM
- Ensure Docker and NVIDIA runtime are installed
- Run using the same `docker compose` commands
- Use mounted persistent disks or buckets for data durability


## Next Steps

- Upload the fine-tuning JSONL to Google Cloud Storage
- Fine-tune a Gemini model using Vertex AI
- Use the FAISS index and Gemini in a custom inference pipeline

## Script Overview

### **parsing_script.py**

Reads the PDF, removes headers and footers, matches section headings, and accumulates section content. Cleans and normalizes text, splits content into fine-tuning and RAG chunks, applies prompt and completion templates, and writes JSONL outputs. Optionally generates embeddings for RAG chunks.

### **setup_vector_db.py**

Reads the RAG JSONL, loads or generates embeddings using a SentenceTransformer (utilizing GPU if available), collects embeddings and metadata, builds a FAISS index, and saves both the index and metadata as a pickle file for fast retrieval.