import { MetadataMode, SimpleVectorStore } from 'llamaindex';
import { UMAP } from 'umap-js';
import { createDocumentStore, createVectorStore } from './embeddings.js';
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
    usedWeaviate: boolean;
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
  docIds: string[];
  config: EmbeddingConfig;
  settings: Settings;
  clients: Clients;
}): Promise<{ embeddings: Map<string, number[]>; metadata: Map<string, any>; usedWeaviate: boolean }> {
  const { docIds, config, settings, clients } = params;
  const embeddings = new Map<string, number[]>();
  const metadata = new Map<string, any>();

  const vectorStore = await createVectorStore(config, settings, clients);
  const bulkResult = await (vectorStore as any).query({docIds: docIds});
  if (bulkResult && typeof bulkResult === 'object') {
    if (bulkResult instanceof Map) {
      for (const [id, vector] of bulkResult.entries()) {
        if (Array.isArray(vector)) embeddings.set(String(id), vector as number[]);
      }
    } else {
      Object.entries(bulkResult as Record<string, unknown>).forEach(([id, vector]) => {
        if (Array.isArray(vector)) embeddings.set(id, vector as number[]);
      });
    }
  }

  if (vectorStore instanceof SimpleVectorStore) {
    const dict = vectorStore.toDict();
    Object.entries(dict.metadataDict ?? {}).forEach(([id, meta]) => {
      metadata.set(id, meta);
    });
  }

  return {
    embeddings,
    metadata,
    usedWeaviate: config.vectorStoreType === 'weaviate',
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
    for (let i = 0; i < 500; i += 1) tsne.step();
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
  const PaCMAP = (mod as any).PaCMAP ?? (mod as any).default ?? mod;
  if (!PaCMAP) throw new Error('PaCMAP module did not export a constructor');

  const nNeighbors = Math.min(50, Math.max(5, embeddings.length - 1));
  const pacmap = new PaCMAP({ nComponents: 2, nNeighbors, distance: 'euclidean' });
  const result = await pacmap.fitTransform(embeddings);
  return (Array.isArray(result) ? result : (result as any).arraySync()) as number[][];
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

  const docStore = await createDocumentStore(config, settings, clients);
  const docsMap = await docStore.docs();

  const nodes = Object.entries(docsMap).map(([id, node]) => ({
    id,
    text: node.getContent ? node.getContent(MetadataMode.NONE) : ((node as any).text ?? ''),
    metadata: node.metadata ?? {},
  }));

  const embeddingSource = await loadEmbeddingsFromVectorStore({
    docIds: nodes.map((node) => node.id),
    config,
    settings,
    clients,
  });

  const available = nodes.filter((node) => embeddingSource.embeddings.has(node.id));
  const missingEmbeddings = nodes.length - available.length;
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
      total: nodes.length,
      missingEmbeddings,
      usedWeaviate: embeddingSource.usedWeaviate,
    },
  };
}
