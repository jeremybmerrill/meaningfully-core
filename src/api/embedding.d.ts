import type { EmbeddingConfig, EmbeddingResult, SearchResult, PreviewResult, Settings, MetadataFilter, Clients } from "../types";
export declare function createEmbeddings(csvPath: string, textColumnName: string, config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<EmbeddingResult>;
export declare function previewResults(csvPath: string, textColumnName: string, config: EmbeddingConfig): Promise<PreviewResult>;
export declare function getDocStore(config: EmbeddingConfig): Promise<import("llamaindex").BaseDocumentStore>;
export declare function getIndex(config: EmbeddingConfig, settings: Settings, clients: Clients): Promise<import("llamaindex").VectorStoreIndex>;
export declare function search(index: any, query: string, numResults?: number, filters?: MetadataFilter[]): Promise<SearchResult[]>;
