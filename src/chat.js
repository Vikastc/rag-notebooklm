import "dotenv/config";
import { OpenAI } from "openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";

const client = new OpenAI();

const CONFIG = {
  PROVIDER: process.env.PROVIDER || "google",
  QDRANT_URL: process.env.QDRANT_URL || "http://localhost:6333",
  DEFAULT_COLLECTION: process.env.DEFAULT_COLLECTION || "web_collection",

  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "text-embedding-3-large",
  CHAT_MODEL: process.env.CHAT_MODEL || "gpt-4o-mini",

  GOOGLE_EMBED_MODEL:
    process.env.GOOGLE_EMBED_MODEL || "models/text-embedding-004",
  GOOGLE_CHAT_MODEL: process.env.GOOGLE_CHAT_MODEL || "gemini-2.5-flash",

  TOP_K: Number(process.env.TOP_K || 3),
  SUBQUERY_COUNT: Number(process.env.SUBQUERY_COUNT || 3),
};

function buildContext(chunks) {
  return chunks
    .map(
      (c, idx) => `
      Chunk ${idx + 1}:
      ${c.pageContent}

      Source: ${c.metadata?.source || c.metadata?.url || "Unknown"}
      ${c.metadata?.title ? `Title: ${c.metadata.title}` : ""}
      `
    )
    .join("\n\n");
}

async function rewriteQuery(query) {
  const rewritePrompt = `
    You are assisting a Retrieval-Augmented Generation (RAG) system. 
    Rewrite the user's query into a clearer, more descriptive form 
    that works better for retrieving relevant chunks from documents. 

    Rules:
    - Do NOT answer the question.
    - Do NOT invent facts.
    - Keep it short, descriptive, and directly tied to the user's intent.

    User query: "${query}"
    Rewritten search query:
  `;

  if (CONFIG.PROVIDER === "google") {
    const chatModel = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      model: CONFIG.GOOGLE_CHAT_MODEL,
    });
    const response = await chatModel.invoke([
      { role: "user", content: rewritePrompt },
    ]);
    return response?.content?.trim() || query;
  } else {
    const response = await client.chat.completions.create({
      model: CONFIG.CHAT_MODEL,
      messages: [{ role: "user", content: rewritePrompt }],
    });
    return response.choices?.[0]?.message?.content?.trim() || query;
  }
}

async function generateSubqueries(rewrittenQuery) {
  const prompt = `
    You are generating retrieval subqueries for a RAG system.

    Task:
    - Break down the following rewritten query into ${CONFIG.SUBQUERY_COUNT} diverse subqueries.
    - Each subquery should capture a slightly different angle or phrasing.
    - Keep them concise.

    Rewritten query: "${rewrittenQuery}"

    Return them as a JSON array of strings.
  `;

  if (CONFIG.PROVIDER === "google") {
    const chatModel = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      model: CONFIG.GOOGLE_CHAT_MODEL,
    });
    const response = await chatModel.invoke([
      { role: "user", content: prompt },
    ]);
    const raw = response?.content?.trim() || "[]";
    try {
      return JSON.parse(raw.match(/\[.*\]/s)?.[0] || "[]");
    } catch {
      return [rewrittenQuery];
    }
  } else {
    const response = await client.chat.completions.create({
      model: CONFIG.CHAT_MODEL,
      messages: [{ role: "user", content: prompt }],
    });
    const raw = response.choices?.[0]?.message?.content?.trim() || "[]";
    try {
      return JSON.parse(raw.match(/\[.*\]/s)?.[0] || "[]");
    } catch {
      return [rewrittenQuery];
    }
  }
}

async function generateHyDEBatch(subqueries) {
  const prompt = `
    You are assisting a Retrieval-Augmented Generation (RAG) system.
    For EACH of the following queries, generate a hypothetical but detailed answer. 
    These answers will NOT be shown to the user—they are only used to enrich embeddings.

    Rules:
    - Write one paragraph per query.
    - Stay factual-sounding but it's okay if details are not exact.
    - Output ONLY a JSON array of strings, each string is one hypothetical answer.

    Queries: ${JSON.stringify(subqueries)}
  `;

  if (CONFIG.PROVIDER === "google") {
    const chatModel = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      model: CONFIG.GOOGLE_CHAT_MODEL,
    });
    const response = await chatModel.invoke([
      { role: "user", content: prompt },
    ]);
    const raw = response?.content?.trim() || "[]";
    try {
      return JSON.parse(raw.match(/\[.*\]/s)?.[0] || "[]");
    } catch {
      return subqueries;
    }
  } else {
    const response = await client.chat.completions.create({
      model: CONFIG.CHAT_MODEL,
      messages: [{ role: "user", content: prompt }],
    });
    const raw = response.choices?.[0]?.message?.content?.trim() || "[]";
    try {
      return JSON.parse(raw.match(/\[.*\]/s)?.[0] || "[]");
    } catch {
      return subqueries;
    }
  }
}

async function retrieveAndRank(vectorStore, subqueries, topK) {
  const retriever = vectorStore.asRetriever({ k: topK });
  const scoreMap = new Map();

  // Step 1: Generate HyDE docs in one go
  const hydeDocs = await generateHyDEBatch(subqueries);

  // Safety: fallback to original subqueries if HyDE fails
  const queriesToRun =
    hydeDocs.length === subqueries.length ? hydeDocs : subqueries;

  // Step 2: Run retrieval in parallel
  const resultsArray = await Promise.all(
    queriesToRun.map((q) => retriever.invoke(q))
  );

  // Step 3: Rank results by vote count
  for (const results of resultsArray) {
    for (const doc of results) {
      const key = doc.id || doc.pageContent.slice(0, 50); // unique-ish key
      const current = scoreMap.get(key) || { doc, score: 0 };
      current.score += 1;
      scoreMap.set(key, current);
    }
  }

  // Step 4: Sort + pick topK
  return [...scoreMap.values()]
    .sort((a, b) => b.score - a.score)
    .map((x) => x.doc)
    .slice(0, topK);
}

// Main chat handler
export async function chatHandler(req, res) {
  try {
    const query = req.body?.query || req.body?.q;
    const collectionName =
      req.body?.collectionName || CONFIG.DEFAULT_COLLECTION;
    const topK = Number(req.body?.k) || CONFIG.TOP_K;
    const history = Array.isArray(req.body?.history) ? req.body.history : [];

    if (!query) {
      return res
        .status(400)
        .json({ error: "Missing 'query' in request body." });
    }

    const embeddings =
      CONFIG.PROVIDER === "google"
        ? new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GOOGLE_API_KEY,
            model: CONFIG.GOOGLE_EMBED_MODEL,
          })
        : new OpenAIEmbeddings({ model: CONFIG.EMBEDDING_MODEL });

    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: CONFIG.QDRANT_URL,
        collectionName,
      }
    );

    // Step 1: rewrite query
    const rewritten = await rewriteQuery(query);
    console.log("✍️ Rewritten query:", rewritten);

    // Step 2: generate subqueries
    const subqueries = await generateSubqueries(rewritten);

    // Step 3: retrieve + rank
    let optimizedChunks = await retrieveAndRank(vectorStore, subqueries, topK);

    if (!optimizedChunks.length) {
      return res.status(200).json({
        answer: "Sorry, I couldn’t find relevant information in the documents.",
        sources: [],
        usedCollection: collectionName,
      });
    }

    // Step 4: Build context and answer
    const contextText = buildContext(optimizedChunks);

    const systemPrompt = `
      You are an AI assistant. 
      Only answer using the context provided. 
      If the answer is not in the context, say: "I couldn't find this in the provided documents."

      Context:
      ${contextText}
    `;

    const messages = [
      { role: "system", content: systemPrompt },
      ...history,
      { role: "user", content: query },
    ];

    let answer = "";
    if (CONFIG.PROVIDER === "google") {
      const chatModel = new ChatGoogleGenerativeAI({
        apiKey: process.env.GOOGLE_API_KEY,
        model: CONFIG.GOOGLE_CHAT_MODEL,
      });
      const response = await chatModel.invoke(messages);
      answer = response?.content || "";
    } else {
      const response = await client.chat.completions.create({
        model: CONFIG.CHAT_MODEL,
        messages,
      });
      answer = response.choices?.[0]?.message?.content || "";
    }

    return res.status(200).json({
      answer,
      sources: optimizedChunks.map((c) => c.metadata),
      usedCollection: collectionName,
      subqueries,
    });
  } catch (err) {
    console.error("🔥 Chat error:", err);
    return res
      .status(500)
      .json({ error: "Chat failed", details: err?.message || String(err) });
  }
}

export default chatHandler;
