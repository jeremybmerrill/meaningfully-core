import type { DocumentSetMetadata, Settings } from './types/index.js';
export declare class DocumentSetManager {
    private sqliteDb;
    constructor(storagePath: string);
    private initializeDatabase;
    addDocumentSet(metadata: Omit<DocumentSetMetadata, 'documentSetId'>): Promise<number>;
    getDocumentSet(documentSetId: number): Promise<DocumentSetMetadata | null>;
    getDocumentSets(page?: number, pageSize?: number): Promise<{
        documents: DocumentSetMetadata[];
        total: number;
    }>;
    updateDocumentCount(documentSetId: number, count: number): Promise<void>;
    deleteDocumentSet(documentSetId: number): Promise<void>;
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
    close(): void;
}
//# sourceMappingURL=DocumentSetManager.d.ts.map