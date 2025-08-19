import "dotenv/config";
import express from "express";
import cors from "cors";
import morgan from "morgan";
import helmet from "helmet";
import multer from "multer";
import os from "os";

import indexingHandler from "./indexing.js";
import chatHandler from "./chat.js";

const app = express();
const PORT = Number(process.env.PORT || 3000);

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

app.listen(PORT, () => {
  console.log(`🚀 Server listening on http://localhost:${PORT}`);
});
