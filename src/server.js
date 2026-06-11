import "dotenv/config";
import express from "express";
import cors from "cors";
import morgan from "morgan";
import helmet from "helmet";
import multer from "multer";
import os from "os";

import indexingHandler from "./indexing.js";
import chatHandler from "./chat.js";

// ── Startup validation ─────────────────────────────────────────────────────
const PROVIDER = (process.env.PROVIDER || "google").trim();

if (!process.env.QDRANT_URL) {
  console.error("❌  Missing QDRANT_URL in .env");
  console.error("    Copy .env.example → .env and fill in your values.");
  process.exit(1);
}

if (PROVIDER === "google") {
  if (!process.env.GOOGLE_API_KEY || process.env.GOOGLE_API_KEY.startsWith("your_")) {
    console.error("❌  Missing or placeholder GOOGLE_API_KEY in .env");
    console.error("    Get your key at: https://aistudio.google.com/app/apikey");
    console.error("    Copy .env.example → .env and fill in your values.");
    process.exit(1);
  }
} else if (PROVIDER === "openai") {
  if (!process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY.startsWith("your_")) {
    console.error("❌  Missing or placeholder OPENAI_API_KEY in .env");
    console.error("    Get your key at: https://platform.openai.com/api-keys");
    console.error("    Copy .env.example → .env and fill in your values.");
    process.exit(1);
  }
} else {
  console.error(`❌  Invalid PROVIDER="${PROVIDER}" in .env  — must be "google" or "openai"`);
  process.exit(1);
}

console.log(`✅  Provider   : ${PROVIDER}`);
console.log(`✅  Qdrant URL : ${process.env.QDRANT_URL}`);
// ──────────────────────────────────────────────────────────────────────────

const app = express();
const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || "127.0.0.1";

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "5mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan("dev"));

// File upload (PDF)
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

// Health
app.get("/health", (_req, res) => res.status(200).json({ ok: true }));

// Indexing endpoints
// - PDF: multipart/form-data with field 'file', optional 'collectionName'
// - URL: JSON body { value: "https://docs...", collectionName? }
app.post("/api/index/:type", upload.single("file"), indexingHandler);

// Chat endpoint: { query, collectionName?, k?, history? }
app.post("/api/chat", chatHandler);

// 404
app.use((req, res) => {
  res.status(404).json({ error: "Not found", path: req.path });
});

// Error handler
app.use((err, _req, res, _next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ error: "Internal Server Error" });
});

app.listen(PORT, HOST, () => {
  console.log(`🚀 Server listening on http://${HOST}:${PORT}`);
});
