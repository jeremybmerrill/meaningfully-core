import { Document, VectorStoreIndex, TextNode, SimpleVectorStore, type StorageContext, BaseDocumentStore, BaseIndexStore } from "llamaindex";
import { OllamaEmbedding } from '@llamaindex/ollama';
import { MistralAIEmbedding } from '@llamaindex/mistral';
import { GeminiEmbedding } from '@llamaindex/google';
import { PGVectorStore } from '@llamaindex/postgres';
import { MockEmbedding } from "./mockEmbedding.js";
import type { EmbeddingConfig, Settings, MetadataFilter, Clients } from "../types/index.js";
import { OpenAIEmbedding } from "@llamaindex/openai";
import { BatchingWeaviateVectorStore } from "./batchingWeaviateVectorStore.js";
import { ProgressVectorStoreIndex } from "./progressVectorStoreIndex.js";
export declare function estimateCost(nodes: TextNode[], modelName: string): {
    estimatedPrice: number;
    tokenCount: number;
    pricePer1M: number;
};
export declare function getExistingVectorStoreIndex(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<VectorStoreIndex>;
export declare function transformDocumentsToNodes(documents: Document[], config: EmbeddingConfig): Promise<TextNode<import("llamaindex").Metadata>[]>;
export declare function getEmbedModel(config: EmbeddingConfig, settings: Settings): OpenAIEmbedding | OllamaEmbedding | MistralAIEmbedding | GeminiEmbedding | MockEmbedding;
export declare function getStorageContext(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<StorageContext>;
export declare function persistDocuments(documents: Document[], config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<void>;
export declare function persistNodes(nodes: TextNode[], config: EmbeddingConfig, settings: Settings, clients: Clients, progressCallback?: (progress: number, total: number) => void): Promise<ProgressVectorStoreIndex>;
export declare function createVectorStore(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<PGVectorStore | SimpleVectorStore | BatchingWeaviateVectorStore>;
export declare function createDocumentStore(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<BaseDocumentStore>;
export declare function createIndexStore(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<BaseIndexStore>;
export declare function searchDocuments(index: VectorStoreIndex, query: string, numResults?: number, filters?: MetadataFilter[]): Promise<import("llamaindex").NodeWithScore<import("llamaindex").Metadata>[]>;
//# sourceMappingURL=embeddings.d.ts.map