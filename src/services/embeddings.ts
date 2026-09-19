import {
  Document,
  VectorStoreIndex,
  // OpenAIEmbedding,
  IngestionPipeline,
  TransformComponent,
  TextNode,
  ModalityType,
  type MetadataFilters,
  type NodeWithScore,
  storageContextFromDefaults,
  SimpleVectorStore,
  type StorageContext,
  Settings as LlamaindexSettings,
  SimpleDocumentStore,
  BaseDocumentStore,
  BaseIndexStore,
  SimpleIndexStore,
  type BaseEmbedding
} from "llamaindex";
import { OllamaEmbedding} from '@llamaindex/ollama'
import { MistralAIEmbedding, MistralAIEmbeddingModelType } from '@llamaindex/mistral'
import { GeminiEmbedding } from '@llamaindex/google'
import { PGVectorStore, PostgresDocumentStore, PostgresIndexStore } from '@llamaindex/postgres';
import { AzureOpenAIEmbedding } from "@llamaindex/azure";
import { Sploder } from "./sploder.js";
import { CustomSentenceSplitter } from "./sentenceSplitter.js";
import { MockEmbedding } from "./mockEmbedding.js";
import { LMStudioEmbedding } from "./lmStudioEmbedding.js";
import { encodingForModel, type TiktokenModel } from "js-tiktoken";
import { join } from "path";
import type { EmbeddingConfig, Settings, MetadataFilter, Clients  } from "../types/index.js";
import { sanitizeProjectName, capitalizeFirstLetter } from "../utils.js";
import * as fs from 'fs';
import { OpenAIEmbedding } from "@llamaindex/openai";
import { BatchingWeaviateVectorStore } from "./batchingWeaviateVectorStore.js";
import { ProgressVectorStoreIndex } from "./progressVectorStoreIndex.js";
import { Bm25Retriever } from "./bm25Retriever.js";

// Used by the postgres vector store, which needs a fixed vector column size up front.
// Models not listed here (e.g. an arbitrary Ollama/LM Studio model) have their dimensions
// determined by actually embedding a probe string -- see getEmbeddingDimensions below --
// since none of these providers' model-listing APIs expose embedding dimensionality.
const MODEL_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-ada-002": 1536,
  "mxbai-embed-large": 1024,
  "mistral-embed": 1024,
  "gemini-embedding-001": 768, // Gemini embedding model
};

// exported only for tests
export async function getEmbeddingDimensions(embeddingModel: BaseEmbedding, modelName: string): Promise<number> {
  const knownDimensions = MODEL_DIMENSIONS[modelName];
  if (knownDimensions) {
    return knownDimensions;
  }
  const probeEmbedding = await embeddingModel.getTextEmbedding("dimension probe");
  return probeEmbedding.length;
}

const PRICE_PER_1M: Record<string, number> = {
  "text-embedding-3-small": 0.02,
  "text-embedding-3-large": 0.13,
  "mistral-embed": 0.1, 
  "mxbai-embed-large": 0, // local model, free
  "nomic-embed-text": 0, // local model, free
  "gemini-embedding-001": 0.0, // Gemini embedding is currently free (unless you're on the paid tier, in which case it is $0.15/million tokens)
};


// Queries a local Ollama instance for installed models, keeping only those that
// support embeddings (per /api/show's "capabilities" field), rather than a hardcoded list.
// See https://github.com/jeremybmerrill/meaningfully/issues/37 and /issues/111
export async function getOllamaEmbeddingModels(baseURL: string): Promise<string[]> {
  const host = baseURL.replace(/\/$/, "");
  const tagsResponse = await fetch(`${host}/api/tags`);
  if (!tagsResponse.ok) {
    throw new Error(`Failed to list Ollama models: ${tagsResponse.status} ${tagsResponse.statusText}`);
  }
  const { models } = (await tagsResponse.json()) as { models: { name: string }[] };

  const embeddingModelNames = await Promise.all(
    models.map(async ({ name }) => {
      const showResponse = await fetch(`${host}/api/show`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: name }),
      });
      if (!showResponse.ok) return null;
      const { capabilities } = (await showResponse.json()) as { capabilities?: string[] };
      return capabilities?.includes("embedding") ? name : null;
    })
  );

  return embeddingModelNames.filter((name): name is string => name !== null);
}

// Queries a local LM Studio instance for installed models, keeping only those whose
// "type" is "embeddings" (LM Studio's own REST API, distinct from its OpenAI-compatible one).
export async function getLMStudioEmbeddingModels(baseURL: string): Promise<string[]> {
  const host = baseURL.replace(/\/$/, "");
  const response = await fetch(`${host}/api/v0/models`);
  if (!response.ok) {
    throw new Error(`Failed to list LM Studio models: ${response.status} ${response.statusText}`);
  }
  const { data } = (await response.json()) as { data: { id: string; type: string }[] };
  return data.filter((model) => model.type === "embeddings").map((model) => model.id);
}

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
  } else if (config.modelProvider === "lmstudio") {
    if (!settings.lmStudioBaseURL) {
      throw new Error("LM Studio base URL is required for LM Studio embedding models");
    }
    embedModel = new LMStudioEmbedding({
      model: config.modelName,
      baseURL: settings.lmStudioBaseURL,
    });
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

// Embedded Weaviate briefly returns "leader not found" while its single-node raft
// group finishes electing a leader after a schema change (e.g. a just-created collection);
// retrying after a short delay is Weaviate's own documented guidance for this transient error.
async function retryOnWeaviateLeaderNotFound<T>(fn: () => Promise<T>, retries = 15, delayMs = 1500): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (retries > 0 && message.includes("leader not found")) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      return retryOnWeaviateLeaderNotFound(fn, retries - 1, delayMs);
    }
    throw error;
  }
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
  const index = await retryOnWeaviateLeaderNotFound(() => ProgressVectorStoreIndex.init({
    nodes: nodes.slice(0, NODE_CHUNK_SIZE),
    storageContext,
    progressCallback: modifiedProgressCallback
  }));
  if (nodes.length > NODE_CHUNK_SIZE) {
    for (let i = NODE_CHUNK_SIZE; i < nodes.length; i += NODE_CHUNK_SIZE) {
      const chunk = nodes.slice(i, i + NODE_CHUNK_SIZE);
      await retryOnWeaviateLeaderNotFound(() => index.insertNodes(chunk, { progressCallback: modifiedProgressCallback }));
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
        dimensions: await getEmbeddingDimensions(embeddingModel, config.modelName),
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
  filters?: MetadataFilter[],
  offset: number = 0
) {
  const safeNumResults = Math.max(1, numResults);
  const safeOffset = Math.max(0, offset);

  const metadataFilters: MetadataFilters = {
    filters: filters ? filters : [],
  };
  const retriever = index.asRetriever({
    similarityTopK: safeOffset + safeNumResults + 1,
    filters: metadataFilters
  });

  const results = (await retriever.retrieve(query)) as NodeWithScore[];
  const page = results.slice(safeOffset, safeOffset + safeNumResults);
  const hasMore = results.length > (safeOffset + safeNumResults);

  return {
    results: page,
    hasMore
  };
}

// Standard damping constant for reciprocal rank fusion (RRF); this is the same default used by
// LlamaIndex Python's QueryFusionRetriever(mode="reciprocal_rerank"), and needs no tuning.
const RRF_K = 60;

// Merges multiple ranked result lists into one, by rank position rather than raw score -- this
// is what makes it possible to combine BM25's keyword score with cosine similarity, which
// aren't on comparable scales. Each occurrence of a node contributes 1/(RRF_K + rank) to its
// fused score; a node appearing near the top of either list scores well.
function reciprocalRankFusion(rankedLists: NodeWithScore[][], k: number = RRF_K): NodeWithScore[] {
  const fusedScores = new Map<string, number>();
  const nodeById = new Map<string, NodeWithScore["node"]>();

  for (const rankedList of rankedLists) {
    rankedList.forEach((result, rank) => {
      const id = result.node.id_;
      nodeById.set(id, result.node);
      fusedScores.set(id, (fusedScores.get(id) ?? 0) + 1 / (k + rank + 1));
    });
  }

  return Array.from(fusedScores.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([id, score]) => ({ node: nodeById.get(id)!, score }));
}

// Hybrid search: fuses LlamaIndexTS's vector-similarity retriever with its Bm25Retriever via
// reciprocal rank fusion, so results benefit from both semantic and exact-keyword matching.
// Bm25Retriever can't apply arbitrary metadata filters (it only takes a list of doc IDs to
// restrict itself to), so when filters are given, the vector leg is queried first (with a
// generous topK) and its matching doc IDs are handed to the BM25 leg -- both legs then rank
// over the same filtered candidate set. Without filters, BM25 scores the whole doc store for
// full recall.
export async function searchDocumentsHybrid(
  index: VectorStoreIndex,
  docStore: BaseDocumentStore,
  query: string,
  numResults: number = 10,
  filters?: MetadataFilter[],
  offset: number = 0
) {
  const safeNumResults = Math.max(1, numResults);
  const safeOffset = Math.max(0, offset);
  const retrievalDepth = safeOffset + safeNumResults + 1;
  const hasFilters = !!filters?.length;

  const metadataFilters: MetadataFilters = {
    filters: filters ? filters : [],
  };

  let vectorResults: NodeWithScore[];
  // The wider, untruncated vector leg (in the filtered branch) -- kept around only so display
  // scores can use a node's real similarity even if it fell outside the RRF fusion depth below.
  let vectorResultsForScoring: NodeWithScore[];
  let bm25Results: NodeWithScore[];
  if (hasFilters) {
    const vectorRetriever = index.asRetriever({
      // Cast a wide net so the BM25 leg (restricted to these same doc IDs, below) sees close to
      // the full set of documents matching the filters, not just the top few by similarity.
      similarityTopK: Math.max(retrievalDepth, 200),
      filters: metadataFilters
    });
    vectorResultsForScoring = (await vectorRetriever.retrieve(query)) as NodeWithScore[];
    const bm25Retriever = new Bm25Retriever({
      docStore,
      topK: retrievalDepth,
      nodes: vectorResultsForScoring.map((result) => result.node),
      docIds: vectorResultsForScoring.map((result) => result.node.id_)
    });
    bm25Results = (await bm25Retriever.retrieve(query)) as NodeWithScore[];
    vectorResults = vectorResultsForScoring.slice(0, retrievalDepth);
  } else {
    const vectorRetriever = index.asRetriever({
      similarityTopK: Math.max(retrievalDepth, 200),
      filters: metadataFilters
    });
    vectorResultsForScoring = (await vectorRetriever.retrieve(query)) as NodeWithScore[];
    const bm25Retriever = new Bm25Retriever({
      docStore,
      topK: retrievalDepth,
      nodes: vectorResultsForScoring.map((result) => result.node),
      docIds: vectorResultsForScoring.map((result) => result.node.id_)
    });
    bm25Results = (await bm25Retriever.retrieve(query)) as NodeWithScore[];
    vectorResults = vectorResultsForScoring.slice(0, retrievalDepth);
  }

  const vectorScoreById = new Map(vectorResultsForScoring.map((result) => [result.node.id_, result.score]));
  bm25Results = bm25Results.filter((result) => vectorScoreById.has(result.node.id_));

  const fused = reciprocalRankFusion([vectorResults, bm25Results]);

  // Copilot says: 
  // The fused RRF score isn't independently interpretable (it's just a rank-based blend), so
  // swap in each result's actual cosine similarity score for display instead -- sourced from the
  // vector leg, which is the only one of the two legs with a comparable score. A node that
  // matched only via BM25 keyword search has no similarity score to show, so it falls back to 0.
  // Note this means displayed scores won't necessarily be monotonically decreasing down the
  // page, since the list is still ordered by the (undisplayed) fused rank, not by this score.
  const fusedWithSimilarityScores = fused.map((result) => ({
    node: result.node,
    score: vectorScoreById.get(result.node.id_) ?? 0
  }));

  const page = fusedWithSimilarityScores.slice(safeOffset, safeOffset + safeNumResults);
  const hasMore = fusedWithSimilarityScores.length > (safeOffset + safeNumResults);

  return {
    results: page,
    hasMore
  };
}
