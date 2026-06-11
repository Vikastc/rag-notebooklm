# 📚 RAG NotebookLM — Indexing & Chat API

A **Retrieval-Augmented Generation (RAG) pipeline** built with **Node.js + LangChain + Qdrant**.
Index documents (PDF, CSV, Websites) into a Qdrant vector database and query them with natural language via a REST API — powered by either **Google Gemini** or **OpenAI**.

🔗 Live Demo: https://verbosity-ai.vercel.app  
▶️ Demo Video: https://youtu.be/_T5p-CFtrtk

---

## 🚀 Features

- 📄 **PDF ingestion** → Upload and index PDF files
- 📊 **CSV ingestion** → Upload and index CSV files
- 🌐 **Website ingestion** → Recursively crawl & index webpages
- 🔎 **Semantic search** → Natural language queries with HyDE + multi-subquery retrieval
- ⚡ **Optimized batching** → Chunking + concurrent inserts to avoid payload limits
- 🤖 **Dual provider support** → Switch between Google Gemini and OpenAI with one env variable

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. Install dependencies

```bash
pnpm install
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

| Variable | Required | Description |
|---|---|---|
| `PROVIDER` | ✅ | `google` (Gemini) or `openai` |
| `GOOGLE_API_KEY` | ✅ if `PROVIDER=google` | [Get from Google AI Studio](https://aistudio.google.com/app/apikey) |
| `OPENAI_API_KEY` | ✅ if `PROVIDER=openai` | [Get from OpenAI Platform](https://platform.openai.com/api-keys) |
| `QDRANT_URL` | ✅ | `http://localhost:6333` for local, or your cloud Qdrant URL |
| `QDRANT_API_KEY` | ⚠️ optional | Only needed if your Qdrant instance requires auth |

> **Both keys are only needed if you switch providers.** Set `PROVIDER=google` and you only need `GOOGLE_API_KEY`. Set `PROVIDER=openai` and you only need `OPENAI_API_KEY`.

### 4. Start Qdrant with Docker Compose

```bash
docker compose up -d qdrant
```

> This spins up Qdrant on `http://localhost:6333`. Skip this step if you're using Qdrant Cloud.

### 5. Start the server

```bash
pnpm dev
```

The server will validate your `.env` on startup and tell you exactly what's missing if anything is wrong.

---

## 📬 API Usage

### 🔹 Index a Website

**POST** `http://localhost:3000/api/index/url`

```json
{
  "value": "https://docs.chaicode.com/",
  "collectionName": "web_collection"
}
```

### 🔹 Index a PDF

**POST** `http://localhost:3000/api/index/pdf`  
Form-data:

| Field | Value |
|---|---|
| `file` | _(PDF file upload)_ |
| `collectionName` | `pdf_collection` |

### 🔹 Index a CSV

**POST** `http://localhost:3000/api/index/csv`  
Form-data:

| Field | Value |
|---|---|
| `file` | _(CSV file upload)_ |
| `collectionName` | `csv_collection` |

### 🔹 Query Indexed Data

**POST** `http://localhost:3000/api/chat`

```json
{
  "query": "What is chaicode?",
  "collectionName": "web_collection",
  "k": 3
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": [...],
  "usedCollection": "web_collection",
  "rewrittenSubqueries": ["...", "..."]
}
```

### 🔹 Health Check

**GET** `http://localhost:3000/health`

---

## 🧪 Postman Collection

Import the included collection to test all endpoints:

👉 [Rag-model.postman_collection.json](./Rag-model.postman_collection.json)

---

## ✅ Example Workflow

1. **Set up your `.env`** with your API key(s) and Qdrant URL.

2. **Start Qdrant locally:**
   ```bash
   docker compose up -d qdrant
   ```

3. **Start the server:**
   ```bash
   pnpm dev
   ```

4. **Index a website:**
   ```bash
   curl -X POST http://localhost:3000/api/index/url \
     -H "Content-Type: application/json" \
     -d '{"value": "https://docs.chaicode.com/", "collectionName": "web_collection"}'
   ```

5. **Query it:**
   ```bash
   curl -X POST http://localhost:3000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What is chaicode?", "collectionName": "web_collection", "k": 3}'
   ```

6. Get AI-powered contextual answers 🎉

---

## 🐳 Run with Docker (full stack)

```bash
cp .env.example .env
# Fill in your API keys in .env, then:
docker compose up -d
```

This runs both Qdrant and the API server.

---

## 🔧 Optional Configuration

All of these have sensible defaults — only override if needed:

```env
# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
BATCH_SIZE=100
CONCURRENCY=10

# Model overrides
GOOGLE_EMBED_MODEL=models/text-embedding-004
GOOGLE_CHAT_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=text-embedding-3-large
CHAT_MODEL=gpt-4o-mini

# Retrieval
TOP_K=3
SUBQUERY_COUNT=2
```
