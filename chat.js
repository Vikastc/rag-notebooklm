import "dotenv/config";
import { OpenAI } from "openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import readline from "readline";

const client = new OpenAI();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function askQuestion(query = "") {
  return new Promise((resolve) => rl.question(query, resolve));
}

async function chat() {
  const userQuery = await askQuestion("Enter your query: ");
  rl.close();
  if (!userQuery) {
    console.error("❌ Query cannot be empty.");
    return;
  }
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
  });

  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: "http://localhost:6333",
      collectionName: "web_collection",
    }
  );

  const vectorRetriever = vectorStore.asRetriever({
    k: 3,
  });

  const relevantChunks = await vectorRetriever.invoke(userQuery);

  const SYSTEM_PROMPT = `You are an AI assistant who fetchs relavant information from the PDF file with 
    the content and page number according to the user query.
    - Only answer from the available context file 

    - example: CORD is a purpose-built decentralised infrastructure designed to be a global public utility and enable a trust framework.
    context
    (Source: Page 23 - 27)

    - example: You can change the default code editor in your system to vscode. To do this, you need to use the following command:
        git config --global core.editor "code --wait"
        (Source: 'https://docs.chaicode.com/youtube/chai-aur-git/terminology/')
            
    Context: ${JSON.stringify(relevantChunks)}
    `;

  const response = await client.chat.completions.create({
    model: "gpt-4",
    messages: [
      {
        role: "system",
        content: SYSTEM_PROMPT,
      },
      {
        role: "user",
        content: userQuery,
      },
    ],
  });

  const assistantReply = response.choices[0].message.content;
  console.log("Assistant Reply> ", assistantReply);
}
// chat();
