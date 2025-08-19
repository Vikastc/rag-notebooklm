# 📚 RAG Indexing & Chat API

This project is a **Retrieval-Augmented Generation (RAG) pipeline** built with **Node.js + LangChain + Qdrant**.
It allows you to **index documents (PDF, CSV, Websites)** into a **Qdrant vector database** and then **query them via an API** using OpenAI embeddings.

---

## 🚀 Features

- 📄 **PDF ingestion** → Upload and index PDF files.
- 📊 **CSV ingestion** → Upload and index CSV files.
- 🌐 **Website ingestion** → Recursively crawl & index webpages.
- 🔎 **Semantic search** → Ask natural language queries and retrieve the most relevant chunks.
- ⚡ **Optimized batching** → Chunking + concurrent inserts to avoid payload limits.

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. Install dependencies with pnpm

```bash
pnpm install
```

### 3. Start Qdrant with Docker Compose

```bash
docker compose up -d
```

> This will spin up Qdrant on `http://localhost:6333`.

### 4. Create a `.env` file

```env
OPENAI_API_KEY=your_openai_api_key

// Optinal
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=   # optional if Qdrant is secured
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
BATCH_SIZE=100
CONCURRENCY=10
```

### 5. Start the server

```bash
pnpm dev
```

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

- `file`: upload a PDF file
- `collectionName`: `pdf_collection`

### 🔹 Index a CSV

**POST** `http://localhost:3000/api/index/csv`
Form-data:

- `file`: upload a CSV file
- `collectionName`: `csv_collection`

### 🔹 Query Indexed Data

**POST** `http://localhost:3000/api/chat`

```json
{
  "query": "Can you go through the csv and explain to me about it?",
  "collectionName": "csv_collection",
  "k": 3
}
```

---

## 🧪 Postman Collection

You can test all endpoints using the provided Postman collection:

👉 [Rag-model.postman_collection.json](./Rag-model.postman_collection.json)

It includes:

- Chat (`/api/chat`)
- Index URL (`/api/index/url`)
- Index PDF (`/api/index/pdf`)
- Index CSV (`/api/index/csv`)

---

## ✅ Example Workflow

1. Spin up Qdrant:

   ```bash
   docker compose up -d
   ```

2. Index a website:

   ```bash
   POST http://localhost:3000/api/index/url
   { "value": "https://docs.chaicode.com/", "collectionName": "web_collection" }
   ```

3. Query it:

   ```bash
   POST http://localhost:3000/api/chat
   { "query": "What is chaicode?", "collectionName": "web_collection", "k": 3 }
   ```

4. Get AI-powered contextual answers 🎉
