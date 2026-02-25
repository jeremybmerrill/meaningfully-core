import { DocumentSetParams, Settings, MetadataFilter, Clients } from './types';
type HasFilePath = {
    filePath: string;
};
type DocumentSetParamsFilePath = DocumentSetParams & HasFilePath;
export declare class MeaningfullyAPI {
    private manager;
    private storagePath;
    private clients;
    constructor({ storagePath, weaviateClient }: {
        storagePath: string;
        weaviateClient?: any;
    });
    setClients(clients: Clients): void;
    getClients(): Clients;
    listDocumentSets(page?: number, pageSize?: number): Promise<{
        documents: import("./types").DocumentSetMetadata[];
        total: number;
    }>;
    getDocumentSet(documentSetId: number): Promise<import("./types").DocumentSetMetadata | null>;
    deleteDocumentSet(documentSetId: number): Promise<{
        success: boolean;
    }>;
    getVectorStoreType(): "simple" | "weaviate";
    generatePreviewData(data: DocumentSetParamsFilePath): Promise<import("./types").PreviewResult>;
    uploadCsv(data: DocumentSetParamsFilePath): Promise<{
        success: boolean;
        documentSetId: number;
    }>;
    searchDocumentSet(documentSetId: number, query: string, n_results?: number, filters?: MetadataFilter[], offset?: number, showContext?: boolean): Promise<import("./types").SearchResponse>;
    getDocument(documentSetId: number, documentNodeId: string): Promise<import("llamaindex").BaseNode<import("llamaindex").Metadata>>;
    getSettings(): Promise<{
        openAIKey: null;
        oLlamaBaseURL: null;
        azureOpenAIKey: null;
        azureOpenAIEndpoint: null;
        azureOpenAIApiVersion: string;
        mistralApiKey: null;
        geminiApiKey: null;
    } & Settings>;
    setSettings(settings: Settings): Promise<Settings & {
        success: boolean;
    }>;
    getMaskedSettings(): Promise<{
        openAIKey: string | null;
        oLlamaBaseURL: null;
        azureOpenAIKey: string | null;
        azureOpenAIEndpoint: null;
        azureOpenAIApiVersion: string;
        mistralApiKey: string | null;
        geminiApiKey: string | null;
    }>;
    setMaskedSettings(newSettings: Settings): Promise<Settings & {
        success: boolean;
    }>;
}
export {};
