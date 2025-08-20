import "dotenv/config";
import { OpenAI } from "openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";

const client = new OpenAI();

const CONFIG = {
  QDRANT_URL: process.env.QDRANT_URL || "http://localhost:6333",
  DEFAULT_COLLECTION: process.env.DEFAULT_COLLECTION || "web_collection",
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "text-embedding-3-large",
  CHAT_MODEL: process.env.CHAT_MODEL || "gpt-4o-mini",
  TOP_K: Number(process.env.TOP_K || 3),
};

function formatSources(chunks) {
  return chunks.map((d, idx) => {
    const meta = d.metadata || {};
    let sourceText = "";

    if (meta.source?.startsWith("http") || meta.url?.startsWith("http")) {
      // Website source
      const url = meta.source || meta.url;
      sourceText = `(Source: '${url}')`;
    } else if (Number.isFinite(meta.page)) {
      // PDF/CSV with page number
      sourceText = `(Source: Page ${meta.page})`;
    }

    return {
      id: idx + 1,
      content: d.pageContent,
      source: sourceText,
      title: meta.title || undefined,
    };
  });
}

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

    const embeddings = new OpenAIEmbeddings({ model: CONFIG.EMBEDDING_MODEL });
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: CONFIG.QDRANT_URL,
        collectionName,
      }
    );

    const retriever = vectorStore.asRetriever({ k: topK });
    const relevantChunks = await retriever.invoke(query);

    const contextText = formatSources(relevantChunks);

    const systemPrompt = `You are an AI assistant who fetchs relavant information from the PDF file with 
    the content and page number according to the user query.
    - Only answer from the available context file 

    // If it is a PDF file or a csv file, then you can use the following format:
    - example_1: CORD is a purpose-built decentralised infrastructure designed to be a global public utility and enable a trust framework.
    context
    (Source: Page 23 - 27)

    // If it is a website, then you can use the following format:
    - example_2: You can change the default code editor in your system to vscode. To do this, you need to use the following command:
        git config --global core.editor "code --wait"
        (Source: 'https://docs.chaicode.com/youtube/chai-aur-git/terminology/')

            
    Context: ${JSON.stringify(contextText)}
    `;

    const messages = [
      { role: "system", content: systemPrompt },
      ...history,
      { role: "user", content: query },
    ];

    const response = await client.chat.completions.create({
      model: CONFIG.CHAT_MODEL,
      messages,
      // temperature: 0.2,
    });

    const answer = response.choices?.[0]?.message?.content || "";
    return res.status(200).json({
      answer,
      sources: relevantChunks[0].metadata,
      usedCollection: collectionName,
    });
  } catch (err) {
    console.error("🔥 Chat error:", err);
    return res
      .status(500)
      .json({ error: "Chat failed", details: err?.message || String(err) });
  }
}

export default chatHandler;
