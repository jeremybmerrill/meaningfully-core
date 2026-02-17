import { MetadataMode, SimpleVectorStore } from 'llamaindex';
import { UMAP } from 'umap-js';
import * as tf from "@tensorflow/tfjs-node";
import { createVectorStore, getExistingVectorStoreIndex, searchDocuments } from './embeddings.js';
import type { Clients, DocumentSetMetadata, EmbeddingConfig, Settings } from '../types/index.js';

export type TopicDefinition = {
  name: string;
  keywords: string[];
  color?: string;
};

export type EmbeddingMapPoint = {
  id: string;
  text: string;
  metadata: Record<string, any>;
  topic: string;
  x: number;
  y: number;
};

export type EmbeddingMapResponse = {
  method: 'pacmap' | 'umap' | 'tsne';
  points: EmbeddingMapPoint[];
  stats: {
    total: number;
    missingEmbeddings: number;
  };
};

function buildEmbeddingConfig(documentSet: DocumentSetMetadata, storagePath: string): EmbeddingConfig {
  const parameters = documentSet.parameters as Record<string, any>;
  return {
    modelName: (parameters.modelName as string) ?? 'text-embedding-3-small',
    modelProvider: (parameters.modelProvider as string) ?? 'openai',
    vectorStoreType: (parameters.vectorStoreType as 'simple' | 'postgres' | 'weaviate') ?? 'simple',
    documentStoreType: (parameters.documentStoreType as 'simple' | 'postgres' | undefined),
    indexStoreType: (parameters.indexStoreType as 'simple' | 'postgres' | undefined),
    projectName: documentSet.name,
    storagePath,
    splitIntoSentences: (parameters.splitIntoSentences as boolean) ?? false,
    combineSentencesIntoChunks: (parameters.combineSentencesIntoChunks as boolean) ?? false,
    sploderMaxSize: (parameters.sploderMaxSize as number) ?? 100,
    chunkSize: (parameters.chunkSize as number) ?? 1024,
    chunkOverlap: (parameters.chunkOverlap as number) ?? 20,
  };
}

async function loadEmbeddingsFromVectorStore(params: {
  maxResults: number;
  config: EmbeddingConfig;
  settings: Settings;
  clients: Clients;
}): Promise<{ embeddings: Map<string, number[]>; metadata: Map<string, any>; nodes: { id: string; text: string; metadata: Record<string, any> }[] }> {
  const { maxResults, config, settings, clients } = params;
  const embeddings = new Map<string, number[]>();
  const metadata = new Map<string, any>();
  const nodes: { id: string; text: string; metadata: Record<string, any> }[] = [];

  const index = await getExistingVectorStoreIndex(config, settings, clients);
  const searchResults = await searchDocuments(index, 'whatever', Math.max(1, maxResults));


  for (const result of searchResults as any[]) {
    const node = result?.node;
    if (!node) continue;

    const id = String(node.id_ ?? node.id ?? '');
    if (!id || embeddings.has(id)) continue;

    const embedding = (node.embedding as number[] | undefined) ?? (typeof node.getEmbedding === 'function' ? node.getEmbedding() : undefined);
    if (!Array.isArray(embedding)) continue;

    const text = node.getContent ? node.getContent(MetadataMode.NONE) : (node.text ?? '');
    const nodeMetadata = (node.metadata ?? {}) as Record<string, any>;
    embeddings.set(id, embedding);
    metadata.set(id, nodeMetadata);
    nodes.push({ id, text, metadata: nodeMetadata });
  }

  const vectorStore = await createVectorStore(config, settings, clients);
  if (vectorStore instanceof SimpleVectorStore) {
    const dict = vectorStore.toDict();
    Object.entries(dict.metadataDict ?? {}).forEach(([id, meta]) => {
      if (!metadata.has(id)) {
        metadata.set(id, meta);
      }
    });
  }

  return {
    embeddings,
    metadata,
    nodes,
  };
}

function classifyTopic(text: string, topics: TopicDefinition[]): string {
  for (const topic of topics) {
    if (!topic.keywords.length) continue;
    const pattern = topic.keywords.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
    const regex = new RegExp(pattern, 'i');
    if (regex.test(text)) return topic.name;
  }
  return 'Uncategorized';
}

async function reduceEmbeddings(embeddings: number[][], method: 'pacmap' | 'umap' | 'tsne') {
  if (!embeddings.length) return [] as number[][];
  if (embeddings.length === 1) return [[0, 0]];

  if (method === 'tsne') {
    // @ts-ignore no type definitions published for tsne-js
    const tsneModule = await import('tsne-js');
    const TSNE = (tsneModule as any).default ?? tsneModule;
    const safePerplexity = Math.min(30, Math.max(2, embeddings.length - 1));
    const tsne = new TSNE({
      dim: 2,
      perplexity: safePerplexity,
      earlyExaggeration: 4.0,
      learningRate: 100,
      nIter: 500,
    });
    tsne.init({ data: embeddings, type: 'dense' });
    tsne.run()
    return tsne.getOutputScaled();
  }

  if (method === 'pacmap') {
    try {
      return await runPacmap(embeddings);
    } catch (error) {
      console.warn('PaCMAP failed; falling back to UMAP', error);
    }
  }

  const umap = new UMAP({
    nComponents: 2,
    nNeighbors: Math.min(15, Math.max(2, embeddings.length - 1)),
    minDist: 0.1,
    random: Math.random,
  });
  return umap.fit(embeddings);
}

async function runPacmap(embeddings: number[][]): Promise<number[][]> {
  // @ts-ignore no type definitions published for pacmap_tfjs
  const mod = await import('pacmap_tfjs');
  const tf = await import('@tensorflow/tfjs-node');
  const PaCMAP = (mod as any).PaCMAP ?? (mod as any).default ?? mod;
  if (!PaCMAP) throw new Error('PaCMAP module did not export a constructor');

  const nNeighbors = Math.min(50, Math.max(5, embeddings.length - 1));
  const pacmap = new PaCMAP({ nComponents: 2, nNeighbors, distance: 'euclidean' });
  
  console.log(embeddings);
  console.log(tf.tensor(embeddings));
  pacmap.fit(tf.tensor(embeddings));
  const ret = await pacmap.Y.array();
  return ret as number[][];
}

export async function generateEmbeddingMap(params: {
  documentSet: DocumentSetMetadata;
  storagePath: string;
  method: 'pacmap' | 'umap' | 'tsne';
  topics?: TopicDefinition[];
  settings: Settings;
  clients: Clients;
}): Promise<EmbeddingMapResponse> {
  const { documentSet, storagePath, method, topics = [], settings, clients } = params;
  const config = buildEmbeddingConfig(documentSet, storagePath);
  const total = Number(documentSet.totalDocuments ?? 0);

  const embeddingSource = await loadEmbeddingsFromVectorStore({
    maxResults: total > 0 ? total : 10000,
    config,
    settings,
    clients,
  });

  const available = embeddingSource.nodes;
  const effectiveTotal = total > 0 ? total : available.length;
  const missingEmbeddings = Math.max(0, effectiveTotal - available.length);
  const embeddingsMatrix = available.map((node) => embeddingSource.embeddings.get(node.id) as number[]);
  const reduced = await reduceEmbeddings(embeddingsMatrix, method);

  const points: EmbeddingMapPoint[] = available.map((node, idx) => ({
    id: node.id,
    text: node.text,
    metadata: {
      ...node.metadata,
      ...(embeddingSource.metadata.get(node.id) ?? {}),
    },
    topic: classifyTopic(node.text ?? '', topics),
    x: reduced[idx]?.[0] ?? 0,
    y: reduced[idx]?.[1] ?? 0,
  }));

  return {
    method,
    points,
    stats: {
      total: effectiveTotal,
      missingEmbeddings,
    },
  };
}
