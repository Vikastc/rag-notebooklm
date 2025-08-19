import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveUrlLoader } from "@langchain/community/document_loaders/web/recursive_url";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import fs from "fs/promises";

/* ---------------- Config ---------------- */
const CONFIG = {
  QDRANT_URL: process.env.QDRANT_URL || "http://localhost:6333",
  QDRANT_API_KEY: process.env.QDRANT_API_KEY || undefined,
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "text-embedding-3-large",
  DEFAULT_PDF_COLLECTION: "pdf_collection",
  DEFAULT_WEB_COLLECTION: "web_collection",
  CHUNK_SIZE: Number(process.env.CHUNK_SIZE || 1000),
  CHUNK_OVERLAP: Number(process.env.CHUNK_OVERLAP || 100),
  BATCH_SIZE: Number(process.env.BATCH_SIZE || 100), // safe size
  CONCURRENCY: Number(process.env.CONCURRENCY || 10), // how many batches to insert in parallel
};

/* ---------------- Helpers ---------------- */
function chunkArray(arr, size) {
  return Array.from({ length: Math.ceil(arr.length / size) }, (_, i) =>
    arr.slice(i * size, i * size + size)
  );
}

function cleanDocuments(docs, fallbackSource = "") {
  return docs.map((d) => {
    const meta = d.metadata || {};
    const outMeta = {};

    if (meta.source) outMeta.source = meta.source;
    if (meta.url) outMeta.url = meta.url;
    if (meta.title) outMeta.title = meta.title;

    const page =
      meta.pageNumber ??
      meta.page ??
      meta.loc?.pageNumber ??
      meta.pdf?.page ??
      meta.pdf?.pagenumber;
    if (Number.isFinite(page)) outMeta.page = page;

    if (!outMeta.source && fallbackSource) outMeta.source = fallbackSource;

    return new Document({
      pageContent: d.pageContent ?? "",
      metadata: outMeta,
    });
  });
}

async function splitDocuments(rawDocs) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CONFIG.CHUNK_SIZE,
    chunkOverlap: CONFIG.CHUNK_OVERLAP,
  });
  return splitter.splitDocuments(rawDocs);
}

/* ---------------- Insert Logic ---------------- */
async function insertInBatches({ docs, embeddings, collectionName }) {
  const batches = chunkArray(docs, CONFIG.BATCH_SIZE);
  if (!batches.length) return;

  console.log(
    `🚀 Inserting ${docs.length} docs in ${batches.length} batches...`
  );

  let vectorStore = null;

  // first batch creates collection
  console.log(
    `📦 Creating collection with first batch (${batches[0].length} docs)...`
  );
  vectorStore = await QdrantVectorStore.fromDocuments(batches[0], embeddings, {
    url: CONFIG.QDRANT_URL,
    apiKey: CONFIG.QDRANT_API_KEY,
    collectionName,
  });

  // remaining batches added concurrently (limited by CONCURRENCY)
  const remaining = batches.slice(1);
  for (let i = 0; i < remaining.length; i += CONFIG.CONCURRENCY) {
    const group = remaining.slice(i, i + CONFIG.CONCURRENCY);
    console.log(`➕ Adding ${group.length} batches in parallel...`);

    await Promise.all(
      group.map((batch, idx) =>
        vectorStore
          .addDocuments(batch)
          .then(() =>
            console.log(
              `   ✅ Batch ${i + idx + 2}/${batches.length} (${
                batch.length
              } docs)`
            )
          )
      )
    );
  }

  console.log(
    `🎉 Done! Inserted ${docs.length} documents into '${collectionName}'`
  );
}

/* ---------------- Loaders ---------------- */
async function loadPDF(filePath) {
  console.log(`📄 Loading PDF: ${filePath}`);
  const rawDocs = await new PDFLoader(filePath).load();
  console.log(`   ✅ Loaded ${rawDocs.length} docs from PDF`);

  const split = await splitDocuments(rawDocs);
  const cleaned = cleanDocuments(split, filePath);
  console.log(`   ✂️ Split into ${cleaned.length} chunks`);
  return cleaned;
}

async function loadWebsite(url) {
  console.log(`🌐 Crawling website: ${url}`);
  const rawDocs = await new RecursiveUrlLoader(url, {
    maxDepth: 2,
    excludeDirs: ["#"],
  }).load();
  console.log(`   ✅ Loaded ${rawDocs.length} docs from website`);

  const split = await splitDocuments(rawDocs);
  const cleaned = cleanDocuments(split, url);
  console.log(`   ✂️ Split into ${cleaned.length} chunks`);
  return cleaned;
}

/* ---------------- API Handler ---------------- */
export async function indexingHandler(req, res) {
  let tempFilePath = null;
  try {
    const argType = (req.params.type || "").toLowerCase(); // "pdf" | "url"
    const providedCollection = req.body?.collectionName;

    if (!["pdf", "url"].includes(argType)) {
      return res
        .status(400)
        .json({ error: "Invalid type. Use 'pdf' or 'url'." });
    }

    const embeddings = new OpenAIEmbeddings({ model: CONFIG.EMBEDDING_MODEL });

    let docs = [];
    let collectionName =
      providedCollection ||
      (argType === "pdf"
        ? CONFIG.DEFAULT_PDF_COLLECTION
        : CONFIG.DEFAULT_WEB_COLLECTION);

    if (argType === "pdf") {
      // Prefer uploaded file if present, otherwise fall back to body.value which should be a server path
      if (req.file?.path) {
        tempFilePath = req.file.path;
        docs = await loadPDF(tempFilePath);
      } else if (req.body?.value) {
        docs = await loadPDF(req.body.value);
      } else {
        return res
          .status(400)
          .json({ error: "Missing PDF file upload or 'value' file path." });
      }
    } else {
      const url = req.body?.value || req.body?.url;
      if (!url) {
        return res
          .status(400)
          .json({ error: "Missing 'value' or 'url' in request body." });
      }
      docs = await loadWebsite(url);
    }

    if (!docs.length) {
      console.warn("⚠️ No documents to insert.");
      return res.status(400).json({ error: "No documents to insert" });
    }

    await insertInBatches({ docs, embeddings, collectionName });

    return res.status(200).json({
      message: "Ingestion complete",
      inserted: docs.length,
      collectionName,
    });
  } catch (err) {
    console.error("🔥 Error in ingestion:", err);
    return res.status(500).json({
      error: "Ingestion failed",
      details: err?.message || String(err),
    });
  } finally {
    if (tempFilePath) {
      try {
        await fs.unlink(tempFilePath);
      } catch {}
    }
  }
}

export default indexingHandler;
