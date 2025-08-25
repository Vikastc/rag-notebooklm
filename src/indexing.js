import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { HtmlToTextTransformer } from "@langchain/community/document_transformers/html_to_text";
import { RecursiveUrlLoader } from "@langchain/community/document_loaders/web/recursive_url";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import fs from "fs/promises";

const CONFIG = {
  QDRANT_URL: process.env.QDRANT_URL || "http://localhost:6333",
  PROVIDER: process.env.PROVIDER || "google",
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "text-embedding-3-large",
  GOOGLE_EMBED_MODEL:
    process.env.GOOGLE_EMBED_MODEL || "models/text-embedding-004",
  DEFAULT_PDF_COLLECTION: "pdf_collection",
  DEFAULT_CSV_COLLECTION: "csv_collection",
  DEFAULT_VTT_COLLECTION: "vtt_collection",
  DEFAULT_WEB_COLLECTION: "web_collection",
  CHUNK_SIZE: Number(process.env.CHUNK_SIZE || 1000),
  CHUNK_OVERLAP: Number(process.env.CHUNK_OVERLAP || 100),
  BATCH_SIZE: Number(process.env.BATCH_SIZE || 100),
  CONCURRENCY: Number(process.env.CONCURRENCY || 10),
};

function chunkArray(arr, size) {
  return Array.from({ length: Math.ceil(arr.length / size) }, (_, i) =>
    arr.slice(i * size, i * size + size)
  );
}

function cleanDocuments(docs, fallbackSource = "") {
  const cleaned = docs
    .map((d) => {
      const meta = d.metadata || {};
      const outMeta = {};

      if (meta.source) outMeta.source = meta.source;
      if (meta.url) outMeta.url = meta.url;
      if (meta.title) outMeta.title = meta.title;

      // Preserve VTT timestamps
      if (meta.startTime) outMeta.startTime = meta.startTime;
      if (meta.endTime) outMeta.endTime = meta.endTime;
      if (meta.type) outMeta.type = meta.type;

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
    })
    .filter((d) => {
      const keep =
        d.pageContent.length > 5 && !d.pageContent.startsWith("Skip to");
      if (!keep) {
        console.log(
          `🚮 Filtering out: "${d.pageContent}" (length: ${d.pageContent.length})`
        );
      }
      return keep;
    });

  console.log(`✅ Kept ${cleaned.length} documents after cleaning`);
  return cleaned;
}

async function splitDocuments(rawDocs) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CONFIG.CHUNK_SIZE,
    chunkOverlap: CONFIG.CHUNK_OVERLAP,
  });
  return splitter.splitDocuments(rawDocs);
}

async function insertInBatches({ docs, embeddings, collectionName }) {
  const batches = chunkArray(docs, CONFIG.BATCH_SIZE);
  if (!batches.length) return;

  console.log(
    `🚀 Inserting ${docs.length} docs in ${batches.length} batches...`
  );

  let vectorStore = null;

  console.log(
    `📦 Creating collection with first batch (${batches[0].length} docs)...`
  );
  vectorStore = await QdrantVectorStore.fromDocuments(batches[0], embeddings, {
    url: CONFIG.QDRANT_URL,
    collectionName,
  });

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

async function loadPDF(filePath) {
  console.log(`📄 Loading PDF: ${filePath}`);
  const rawDocs = await new PDFLoader(filePath).load();
  const split = await splitDocuments(rawDocs);
  return cleanDocuments(split, filePath);
}

async function loadCSV(filePath) {
  console.log(`📄 Loading CSV: ${filePath}`);
  const rawDocs = await new CSVLoader(filePath).load();
  const split = await splitDocuments(rawDocs);
  return cleanDocuments(split, filePath);
}

async function parseVTTContent(content, filePath) {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line);
  const cues = [];
  let i = 0;

  // Skip WEBVTT header
  while (i < lines.length && !lines[i].includes("-->")) {
    i++;
  }

  while (i < lines.length) {
    const line = lines[i];

    // Check if this line is a timestamp
    if (line && line.includes("-->")) {
      const [startTime, endTime] = line.split("-->").map((t) => t.trim());

      // Collect text lines after timestamp
      const textLines = [];
      i++;
      while (i < lines.length && !lines[i].includes("-->")) {
        textLines.push(lines[i]);
        i++;
      }

      if (textLines.length > 0) {
        const text = textLines
          .join(" ")
          .replace(/<[^>]*>/g, "")
          .trim();
        if (text.length > 0) {
          cues.push({
            startTime,
            endTime,
            text,
          });
        }
      }
    } else {
      i++;
    }
  }

  return cues.map(
    (cue) =>
      new Document({
        pageContent: cue.text,
        metadata: {
          source: filePath,
          startTime: cue.startTime,
          endTime: cue.endTime,
          type: "vtt",
        },
      })
  );
}

async function loadVTT(filePath) {
  console.log(`📄 Loading VTT: ${filePath}`);
  const content = await fs.readFile(filePath, "utf-8");
  const docs = await parseVTTContent(content, filePath);

  return cleanDocuments(docs, filePath);
}

async function loadWebsite(url) {
  console.log(`🌐 Crawling website: ${url}`);
  const rawDocs = await new RecursiveUrlLoader(url, {
    maxDepth: 2,
    excludeDirs: ["#"],
  }).load();

  const transformer = new HtmlToTextTransformer();
  const textDocs = await transformer.transformDocuments(rawDocs);

  const split = await splitDocuments(textDocs);
  return cleanDocuments(split, url);
}

// Api Handlers
export async function indexingHandler(req, res) {
  let tempFilePath = null;
  try {
    const argType = (req.params.type || "").toLowerCase();
    const providedCollection = req.body?.collectionName;

    if (!["pdf", "csv", "vtt", "url"].includes(argType)) {
      return res.status(400).json({
        error: "Invalid type. Use 'pdf', 'csv', 'vtt', or 'url'.",
      });
    }

    const embeddings =
      CONFIG.PROVIDER === "google"
        ? new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GOOGLE_API_KEY,
            model: CONFIG.GOOGLE_EMBED_MODEL,
          })
        : new OpenAIEmbeddings({ model: CONFIG.EMBEDDING_MODEL });

    let docs = [];
    let collectionName =
      providedCollection ||
      (argType === "pdf"
        ? CONFIG.DEFAULT_PDF_COLLECTION
        : argType === "csv"
        ? CONFIG.DEFAULT_CSV_COLLECTION
        : argType === "vtt"
        ? CONFIG.DEFAULT_VTT_COLLECTION
        : CONFIG.DEFAULT_WEB_COLLECTION);

    if (argType === "pdf") {
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
    } else if (argType === "csv") {
      if (req.file?.path) {
        tempFilePath = req.file.path;
        docs = await loadCSV(tempFilePath);
      } else if (req.body?.value) {
        docs = await loadCSV(req.body.value);
      } else {
        return res
          .status(400)
          .json({ error: "Missing CSV file upload or 'value' file path." });
      }
    } else if (argType === "vtt") {
      if (req.file?.path) {
        tempFilePath = req.file.path;
        docs = await loadVTT(tempFilePath);
      } else if (req.body?.value) {
        docs = await loadVTT(req.body.value);
      } else {
        return res
          .status(400)
          .json({ error: "Missing VTT file upload or 'value' file path." });
      }
    } else if (argType === "url") {
      const url = req.body?.value || req.body?.url;
      if (!url) {
        return res
          .status(400)
          .json({ error: "Missing 'value' or 'url' in request body." });
      }
      docs = await loadWebsite(url);
    }

    if (!docs.length) {
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
