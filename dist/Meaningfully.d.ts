import { MetadataManager } from './MetadataManager.js';
import type { DocumentSetParams, Settings, MetadataFilter, Clients } from './types/index.js';
type HasFilePath = {
    filePath: string;
};
type DocumentSetParamsFilePath = DocumentSetParams & HasFilePath;
export declare class MeaningfullyAPI {
    private manager;
    private storagePath;
    private clients;
    constructor({ storagePath, weaviateClient, metadataManager }: {
        storagePath: string;
        weaviateClient?: any;
        metadataManager: MetadataManager;
    });
    setClients(clients: Clients): void;
    getClients(): Clients;
    listDocumentSets(page?: number, pageSize?: number): Promise<{
        documents: import("./types/index.js").DocumentSetMetadata[];
        total: number;
    }>;
    getDocumentSet(documentSetId: number): Promise<import("./types/index.js").DocumentSetMetadata | null>;
    deleteDocumentSet(documentSetId: number): Promise<{
        success: boolean;
    }>;
    getVectorStoreType(): "weaviate" | "simple";
    generatePreviewData(data: DocumentSetParamsFilePath): Promise<import("./types/index.js").PreviewResult>;
    uploadCsv(data: DocumentSetParamsFilePath): Promise<{
        success: boolean;
        documentSetId: number;
    }>;
    searchDocumentSet(documentSetId: number, query: string, n_results?: number, filters?: MetadataFilter[]): Promise<import("./types/index.js").SearchResult[]>;
    getDocument(documentSetId: number, documentNodeId: string): Promise<import("llamaindex").BaseNode<import("llamaindex").Metadata>>;
    getSettings(): Promise<Settings>;
    setSettings(settings: Settings): Promise<{
        success: boolean;
    }>;
    getMaskedSettings(): Promise<{
        openAIKey: string | null;
        oLlamaBaseURL: string | null;
        azureOpenAIKey: string | null;
        azureOpenAIEndpoint: string | null;
        azureOpenAIApiVersion: string | null;
        mistralApiKey: string | null;
        geminiApiKey: string | null;
    }>;
    setMaskedSettings(newSettings: Settings): Promise<{
        success: boolean;
    }>;
}
export {};
//# sourceMappingURL=Meaningfully.d.ts.map