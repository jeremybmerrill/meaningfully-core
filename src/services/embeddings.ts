import { 
  Document, 
  VectorStoreIndex, 
  // OpenAIEmbedding,
  IngestionPipeline,
  TransformComponent,
  TextNode,
  ModalityType,
  type MetadataFilters,
  storageContextFromDefaults,
  SimpleVectorStore,
  type StorageContext,
  Settings as LlamaindexSettings,
  SimpleDocumentStore,
  BaseDocumentStore,
  BaseIndexStore,
  SimpleIndexStore
} from "llamaindex";
import { OllamaEmbedding} from '@llamaindex/ollama'
import { MistralAIEmbedding, MistralAIEmbeddingModelType } from '@llamaindex/mistral'
import { GeminiEmbedding } from '@llamaindex/google'
import { PGVectorStore, PostgresDocumentStore, PostgresIndexStore } from '@llamaindex/postgres';
import { AzureOpenAIEmbedding } from "@llamaindex/azure";
import { Sploder } from "./sploder.js";
import { CustomSentenceSplitter } from "./sentenceSplitter.js";
import { MockEmbedding } from "./mockEmbedding.js";
import { encodingForModel, type TiktokenModel } from "js-tiktoken";
import { join } from "path";
import type { EmbeddingConfig, Settings, MetadataFilter, Clients  } from "../types/index.js";
import { sanitizeProjectName, capitalizeFirstLetter } from "../utils.js";
import * as fs from 'fs';
import { OpenAIEmbedding } from "@llamaindex/openai";
import { BatchingWeaviateVectorStore } from "./batchingWeaviateVectorStore.js";
import { ProgressVectorStoreIndex } from "./progressVectorStoreIndex.js";

// unused, but probalby eventually will be used.
// to be used by postgres store, which it' slooking increasingly like I have to enable again
const MODEL_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "mxbai-embed-large": 1024,
  "mistral-embed": 1024,
  "gemini-embedding-001": 768, // Gemini embedding model
};

const PRICE_PER_1M: Record<string, number> = {
  "text-embedding-3-small": 0.02,
  "text-embedding-3-large": 0.13,
  "mistral-embed": 0.1, 
  "mxbai-embed-large": 0, // local model, free
  "nomic-embed-text": 0, // local model, free
  "gemini-embedding-001": 0.0, // Gemini embedding is currently free (unless you're on the paid tier, in which case it is $0.15/million tokens)
};


/* all transformations except the embedding step (which is handled by VectorStoreIndex.init) */
function getBaseTransformations(config: EmbeddingConfig){
  const transformations: TransformComponent[] = [
    new CustomSentenceSplitter({ chunkSize: config.chunkSize, chunkOverlap: config.chunkOverlap }),
  ];

  if (config.combineSentencesIntoChunks) {
    transformations.push(
      new Sploder({
        maxStringTokenCount: config.sploderMaxSize
      })
    );
  }

  return transformations;
}

export function estimateCost(nodes: TextNode[], modelName: string): {
  estimatedPrice: number;
  tokenCount: number;
  pricePer1M: number;
} {
  const pricePer1M = PRICE_PER_1M[modelName] || 0; // default to 0 if model not found or free

  let tokenizer; 
  try{
    tokenizer = encodingForModel(modelName as TiktokenModel); // This doesn't work for ollama
  } catch (error) {
    // If the tokenizer is not found, it means the model is likely not supported by tiktoken
    // or is a local model (like Ollama). In this case, we can't estimate the cost.
    tokenizer = encodingForModel("text-embedding-3-small"); // fallback to a known tokenizer
    console.warn(`Tokenizer for model ${modelName} not found. Using fallback tokenizer.`);
  }
  const tokenCount = nodes.reduce((sum, node) => {
    return sum + tokenizer.encode(node.text).length;
  }, 0);

  const estimatedPrice = tokenCount * (pricePer1M / 1_000_000);

  return {
    estimatedPrice,
    tokenCount,
    pricePer1M
  };
}

export async function getExistingVectorStoreIndex(config: EmbeddingConfig, settings: Settings, clients: Clients) {
  let storageContext: StorageContext;
  switch (config.vectorStoreType) {
    case "simple":
      const embedModel = getEmbedModel(config, settings);      
      const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName));
      storageContext = await storageContextFromDefaults({
        persistDir: persistDir,
      });
      let vsi = await VectorStoreIndex.init({
        storageContext: storageContext,
      });
      vsi.embedModel = embedModel;
      return vsi;
    default:
      storageContext = await getStorageContext(config, settings, clients);
      const vectorStore = storageContext.vectorStores[ModalityType.TEXT];
      if (!vectorStore) {
        throw new Error("Vector store for ModalityType.TEXT is undefined");
      }
      return await VectorStoreIndex.fromVectorStore(vectorStore);
  }
}

export async function transformDocumentsToNodes(
  documents: Document[],
  config: EmbeddingConfig,
) {
  console.time("transformDocumentsToNodes Run Time");

  const transformations = getBaseTransformations(config);

  // llama-index stupidly includes all the metadata in the embedding, which is a waste of tokens
  // so we exclude everything except the text column from the embedding
  for (const document of documents) {
    document.excludedEmbedMetadataKeys = Object.keys(document.metadata);
  }
  console.time("transformDocumentsToNodes transformDocuments Run Time");
  // remove empty documents. we can't meaningfully embed these, so we're just gonna ignore 'em.
  // that might not ultimately be the right solution. 
  documents = documents.filter((document_) => document_.text && document_.text.length > 0);

  // Create nodes with sentence splitting and optional sploder
  const pipeline = new IngestionPipeline({
    transformations
  });

  const nodes = (await pipeline.run({documents: documents})) as TextNode[];

  console.timeEnd("transformDocumentsToNodes transformDocuments Run Time");
  console.timeEnd("transformDocumentsToNodes Run Time");  
  return nodes;
}

export function getEmbedModel(
  config: EmbeddingConfig, 
  settings: Settings,
) {
  let embedModel; 
  if (config.modelProvider === "openai" ){
    embedModel = new OpenAIEmbedding({ model: config.modelName, apiKey: settings.openAIKey ? settings.openAIKey : undefined} );
    embedModel.embedBatchSize = 50; // all embedding models enforce a maximum of 300,000 tokens summed across all inputs in a single request
  } else if (config.modelProvider === "ollama") {
    embedModel = new OllamaEmbedding({ model: config.modelName, config: {
      host: settings.oLlamaBaseURL ? settings.oLlamaBaseURL : undefined
    }, }); 
  } else if (config.modelProvider === "azure") {
    if (!settings.azureOpenAIKey || !settings.azureOpenAIEndpoint) {
      throw new Error("Azure OpenAI API key and endpoint are required for Azure embedding models");
    }
    embedModel = new AzureOpenAIEmbedding({ 
      model: config.modelName, 
      apiKey: settings.azureOpenAIKey,
      endpoint: settings.azureOpenAIEndpoint,
      apiVersion: settings.azureOpenAIApiVersion ?? undefined
    });
  } else if (config.modelProvider === "mistral") {
    if (!settings.mistralApiKey) {
      throw new Error("Mistral API key is required for Mistral embedding models");
    }
    embedModel = new MistralAIEmbedding({ 
      model: MistralAIEmbeddingModelType.MISTRAL_EMBED, // only one choice!
      apiKey: settings.mistralApiKey
    });
  } else if (config.modelProvider === "gemini") {
    if (!settings.geminiApiKey) {
      throw new Error("Gemini API key is required for Gemini embedding models");
    }
    embedModel = new GeminiEmbedding({ 
      apiKey: settings.geminiApiKey,
    });
    embedModel.embedBatchSize = 50;
  } else if (config.modelProvider === "mock") {
    embedModel = new MockEmbedding();
  } else {
    throw new Error(`Unsupported embedding model provider: ${config.modelProvider}`);
  }
  LlamaindexSettings.embedModel = embedModel;
  return embedModel;
}

export async function getStorageContext(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<StorageContext> {
  const vectorStore = await createVectorStore(config, settings, clients);
  const docStore = await createDocumentStore(config, settings, clients); // new SimpleDocumentStore()
  const indexStore = await createIndexStore(config, settings, clients);
  fs.mkdirSync(config.storagePath, { recursive: true }); 
  const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName) );
  return await storageContextFromDefaults({
    persistDir: persistDir,
    vectorStores: {[ModalityType.TEXT]: vectorStore},
    docStore: docStore,
    indexStore: indexStore
      /*
        if docStore is created with a persist path (as it is by default in storageContextFromDefaults)
        then it will write to disk after every put(), which happens 2+ times per document.

        so we create it without a persist path, and then explicitly persist it when we're done adding documents.

        see https://github.com/jeremybmerrill/meaningfully/issues/52
      */
  });
}

export async function persistDocuments(documents: Document[], config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<void> {
  console.time("persistDocuments Run Time");
  const storageContext = await getStorageContext(config, settings, clients);
  await storageContext.docStore.addDocuments(documents, true);

  // see comments in getStorageContext
  const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName) );
  if (storageContext.docStore instanceof SimpleDocumentStore) {
    // @ts-ignore
    await (storageContext.docStore as SimpleDocumentStore).kvStore.persist(join(persistDir, "doc_store.json"));
  }else if (storageContext.docStore instanceof PostgresDocumentStore) {
    // PostgresDocumentStore does not need to be explicitly persisted, so we don't include it in the OR conditional here..
    console.log("Pretending to persist Postgres document store, but it actually persists automatically.");
  }

  console.timeEnd("persistDocuments Run Time");
}

export async function persistNodes(nodes: TextNode[], config: EmbeddingConfig, settings: Settings, clients: Clients, progressCallback?: (progress: number, total: number) => void): Promise<ProgressVectorStoreIndex> {
  // Create and configure vector store based on type
  console.time("persistNodes Run Time");

  const storageContext = await getStorageContext(config, settings, clients);
  const vectorStore = storageContext.vectorStores[ModalityType.TEXT];
  if (!vectorStore) {
    throw new Error("Vector store is undefined");
  }
  // Create index and embed documents
  // this is what actaully embeds the nodes
  // (even if they already have embeddings, stupidly)
  const NODE_CHUNK_SIZE = 10000
  const modifiedProgressCallback = (progress: number, total: number) => {
    if (progressCallback) {
      progressCallback(progress, nodes.length)
    }
    console.log('progress total nodes.length', progress, nodes.length); // TODO: give this `i`, so that it knows how many nodes have been processed so far outside of this chunk.
  }
  const index = await ProgressVectorStoreIndex.init({
    nodes: nodes.slice(0, NODE_CHUNK_SIZE), 
    storageContext,
    progressCallback: modifiedProgressCallback
  });
  if (nodes.length > NODE_CHUNK_SIZE) {
    for (let i = NODE_CHUNK_SIZE; i < nodes.length; i += NODE_CHUNK_SIZE) {
      const chunk = nodes.slice(i, i + NODE_CHUNK_SIZE);
      await index.insertNodes(chunk, { progressCallback: modifiedProgressCallback });
    }
  }

  // I'm not sure why this explicit call to persist is necessary. 
  // storageContext should handle this, but it doesn't.
  // all the if statements are just type-checking boilerplate.
  // N.B. WeaviateVectorStore does not need to be explicitly persisted, so we don't include it in the OR conditional here..
  if (vectorStore) {
    if (vectorStore instanceof SimpleVectorStore) {
      await vectorStore.persist(join(config.storagePath, sanitizeProjectName(config.projectName), "vector_store.json"));
    } else if (vectorStore instanceof PGVectorStore || vectorStore instanceof BatchingWeaviateVectorStore) {
      // WeaviateVectorStore does not have a persist method, it persists automatically
      console.log("Pretending to persist Weaviate or Postgres vector store, but it actually persists automatically.");
    } else {
      throw new Error("Vector store does not support persist method");
    }
  } else {
    throw new Error("Vector store is undefined");
  }
  console.timeEnd("persistNodes Run Time");
  return index;
}

// exported only for tests
export async function createVectorStore(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<PGVectorStore | SimpleVectorStore | BatchingWeaviateVectorStore> {
  const embeddingModel = getEmbedModel(config, settings);
  switch (config.vectorStoreType) {

    // for some reason the embedding model has to be specified here TOO
    // otherwise it defaults to Ada.
    case "postgres":
      return new PGVectorStore({
        client: clients.postgresClient,
        tableName: "vecs_" + sanitizeProjectName(config.projectName),
        dimensions: MODEL_DIMENSIONS[config.modelName] || 1536, // default to 1536 if model not found
        embeddingModel: embeddingModel
      });

    case "simple":
      const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName));
      return SimpleVectorStore.fromPersistDir(persistDir, embeddingModel);

    case "weaviate": 
      const vectorStore = new BatchingWeaviateVectorStore({
        indexName: capitalizeFirstLetter(sanitizeProjectName(config.projectName)), 
        weaviateClient: clients.weaviateClient, 
        embeddingModel: embeddingModel 
      });

      // WeaviateVectorStore's getNodeSimilarity method looks for distance, but current weaviate provides score
      // (WeaviateVectorStore would get `score` if we were doing hybrid search)
      // Overwrite the private getNodeSimilarity method to use 'score' from metadata
      // @ts-ignore
      vectorStore.getNodeSimilarity = (entry, _similarityKey = "score") => {
        return  entry.metadata.score;
      }

      return vectorStore;
    default:
      throw new Error(`Unsupported vector store type: ${config.vectorStoreType}`);
  }
}

// exported only for tests
export async function createDocumentStore(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<BaseDocumentStore> {
  // we create the doc store without a persist path, so it doesn't write to disk after every put()
  switch (config.documentStoreType || config.vectorStoreType) {
    case "postgres":
      return new PostgresDocumentStore({
        client: clients.postgresClient,
        tableName: "docs_" + sanitizeProjectName(config.projectName),
      });
    case "simple":
    case "weaviate": 
      const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName));
      return SimpleDocumentStore.fromPersistDir(persistDir);
    default:
      throw new Error(`Unsupported vector store type: ${config.vectorStoreType}`);
  }
}

// exported only for tests
export async function createIndexStore(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<BaseIndexStore> {
  switch (config.documentStoreType || config.vectorStoreType) {
    case "postgres":
      return new PostgresIndexStore({
        client: clients.postgresClient,
        tableName: "idx_" + sanitizeProjectName(config.projectName),
      });
    case "simple":
    case "weaviate": 
      const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName));
      return SimpleIndexStore.fromPersistDir(persistDir);
    default:
      throw new Error(`Unsupported vector store type: ${config.vectorStoreType}`);
  }

}

export async function searchDocuments(
  index: VectorStoreIndex,
  query: string,
  numResults: number = 10,
  filters?: MetadataFilter[]
) {
  // const metadataFilters: MetadataFilters | undefined = filters ? {filters: filters} : undefined;
  const metadataFilters: MetadataFilters = {
    filters: filters ? filters : [],
  };
  const retriever = index.asRetriever({ similarityTopK: numResults, filters: metadataFilters });

  const results = await retriever.retrieve(query );
  return results;
}
