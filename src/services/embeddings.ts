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
  SimpleDocumentStore
} from "llamaindex";
import { OllamaEmbedding} from '@llamaindex/ollama'
import { MistralAIEmbedding, MistralAIEmbeddingModelType } from '@llamaindex/mistral'
import { GeminiEmbedding } from '@llamaindex/google'
import { PGVectorStore, PostgresDocumentStore } from '@llamaindex/postgres';
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
  const embedModel = getEmbedModel(config, settings);
  switch (config.vectorStoreType) {
    case "simple":
      const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName));
      const storageContext = await storageContextFromDefaults({
        persistDir: persistDir,
      });
      let vsi = await VectorStoreIndex.init({
        storageContext: storageContext,
      });
      vsi.embedModel = embedModel;
      return vsi;

    case "postgres":
      if (!clients.postgresClient) {
        throw new Error("Postgres client required but not provided");
      }
      const pgStore = new PGVectorStore({
        client: clients.postgresClient,
        tableName: sanitizeProjectName(config.projectName),
        dimensions: MODEL_DIMENSIONS[config.modelName] || 1536, // default to 1536 if model not found
        embeddingModel: embedModel
      });
      const pgStorageContext = await storageContextFromDefaults({
        vectorStores: { [ModalityType.TEXT]: pgStore },
      });
      return await VectorStoreIndex.init({
        storageContext: pgStorageContext,
      });
    case "weaviate":
      if (!clients.weaviateClient) {
        throw new Error("Weaviate client required but not provided");
      }
      const weaviateStore = new BatchingWeaviateVectorStore({
        indexName: capitalizeFirstLetter(sanitizeProjectName(config.projectName)),
        weaviateClient: clients.weaviateClient,
        embeddingModel: embedModel
      });

      // WeaviateVectorStore's getNodeSimilarity method looks for distance, but current weaviate provides score
      // (WeaviateVectorStore would get `score` if we were doing hybrid search)
      // Overwrite the private getNodeSimilarity method to use 'score' from metadata
      // @ts-ignore
      weaviateStore.getNodeSimilarity = (entry, _similarityKey = "score") => {
        return  entry.metadata.score;
      }

      return await VectorStoreIndex.fromVectorStore(weaviateStore)

    default:
      throw new Error(`Unsupported vector store type: ${config.vectorStoreType}`);
  }
}

export async function getExistingDocStore(config: EmbeddingConfig) {
  switch (config.vectorStoreType) {
    case "simple":
    case "weaviate":
      // both simple and weaviate use the same doc store type
      // so we can just use the simple doc store for both
      const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName) );
      const storageContext = await storageContextFromDefaults({
        persistDir: persistDir,
      });
      return storageContext.docStore;
    case "postgres":
      throw new Error(`Not yet implemented vector store type: ${config.vectorStoreType}`);
      // return await createVectorStore(config);
    default:
      throw new Error(`Unsupported vector store type: ${config.vectorStoreType}`);
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
  fs.mkdirSync(config.storagePath, { recursive: true }); 
  const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName) );
  switch (config.vectorStoreType) {
    case "simple":
    case "weaviate":
      // both simple and weaviate use the same doc store type
      // so we can just use the simple doc store for both
      return await storageContextFromDefaults({
        persistDir: persistDir,
        vectorStores: {[ModalityType.TEXT]: vectorStore},
        docStore: new SimpleDocumentStore()
        /*
          if docStore is created with a persist path (as it is by default in storageContextFromDefaults)
          then it will write to disk after every put(), which happens 2+ times per document.

          so we create it without a persist path, and then explicitly persist it when we're done adding documents.

          see https://github.com/jeremybmerrill/meaningfully/issues/52
        */
      });
    case "postgres":
      if (!clients.postgresClient) {
        throw new Error("Postgres client required but not provided");
      }
      return await storageContextFromDefaults({
        vectorStores: {[ModalityType.TEXT]: vectorStore},
        docStore: new PostgresDocumentStore({
          client: clients.postgresClient,
          tableName: "docs_" + sanitizeProjectName(config.projectName),
        }) 
        /*
          PostgresDocumentStore does not have a persist method, it persists automatically
        */
      });
    default: 
      throw new Error(`Unsupported vector store type: ${config.vectorStoreType}`);
    
  }
}

export async function persistDocuments(documents: Document[], config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<void> {
  console.time("persistDocuments Run Time");
  const storageContext = await getStorageContext(config, settings, clients);
  await storageContext.docStore.addDocuments(documents, true);

  // see comments in getStorageContext
  const persistDir = join(config.storagePath, sanitizeProjectName(config.projectName) );
  // @ts-ignore
  await (storageContext.docStore as SimpleDocumentStore).kvStore.persist(join(persistDir, "doc_store.json"));

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
  const index = await ProgressVectorStoreIndex.init({
    nodes, 
    storageContext, 
    logProgress: true,
    progressCallback,
  });

  // I'm not sure why this explicit call to persist is necessary. 
  // storageContext should handle this, but it doesn't.
  // all the if statements are just type-checking boilerplate.
  // N.B. WeaviateVectorStore does not need to be explicitly persisted, so we don't include it in the OR conditional here..
  if (vectorStore) {
    if (vectorStore instanceof PGVectorStore || vectorStore instanceof SimpleVectorStore) {
      await vectorStore.persist(join(config.storagePath, sanitizeProjectName(config.projectName), "vector_store.json"));
    } else if (vectorStore instanceof BatchingWeaviateVectorStore) {
      // WeaviateVectorStore does not have a persist method, it persists automatically
      console.log("Pretending to persist Weaviate vector store, but it actually persists automatically.");
    } else {
      throw new Error("Vector store does not support persist method");
    }
  } else {
    throw new Error("Vector store is undefined");
  }
  console.timeEnd("persistNodes Run Time");
  return index;
}

async function createVectorStore(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<PGVectorStore | SimpleVectorStore | BatchingWeaviateVectorStore> {
  const embeddingModel = getEmbedModel(config, settings);
  switch (config.vectorStoreType) {

    // for some reason the embedding model has to be specified here TOO
    // otherwise it defaults to Ada.
    case "postgres":
      return new PGVectorStore({
        clientConfig: {connectionString: process.env.POSTGRES_CONNECTION_STRING},
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
