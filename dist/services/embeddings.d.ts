import { Document, VectorStoreIndex, TextNode, type StorageContext } from "llamaindex";
import { OllamaEmbedding } from '@llamaindex/ollama';
import { MistralAIEmbedding } from '@llamaindex/mistral';
import { GeminiEmbedding } from '@llamaindex/google';
import { MockEmbedding } from "./mockEmbedding.js";
import type { EmbeddingConfig, Settings, MetadataFilter, Clients } from "../types/index.js";
import { OpenAIEmbedding } from "@llamaindex/openai";
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
export declare function searchDocuments(index: VectorStoreIndex, query: string, numResults?: number, filters?: MetadataFilter[]): Promise<import("llamaindex").NodeWithScore<import("llamaindex").Metadata>[]>;
//# sourceMappingURL=embeddings.d.ts.map