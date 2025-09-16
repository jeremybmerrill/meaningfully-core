import type { DocumentSetMetadata, Settings } from './types/index.js';
export declare abstract class MetadataManager {
    protected queries: {
        createDocumentSetsTable: string;
        createSettingsTable: string;
        insertDocumentSet: string;
        selectDocumentSet: string;
        selectDocumentSets: string;
        countDocumentSets: string;
        updateDocumentCount: string;
        deleteDocumentSet: string;
        selectSettings: string;
        upsertSettings: string;
    };
    protected abstract runQuery<T>(query: string, params?: any[]): Promise<T[]>;
    protected abstract runQuerySingle<T>(query: string, params?: any[]): Promise<T | null>;
    protected abstract initializeDatabase(): Promise<void>;
    protected abstract close(): void;
    addDocumentSet(metadata: Omit<DocumentSetMetadata, 'documentSetId'>): Promise<number>;
    getDocumentSet(documentSetId: number): Promise<DocumentSetMetadata | null>;
    getDocumentSets(page?: number, pageSize?: number): Promise<{
        documents: DocumentSetMetadata[];
        total: number;
    }>;
    updateDocumentCount(documentSetId: number, count: number): Promise<void>;
    deleteDocumentSet(documentSetId: number): Promise<void>;
    getSettings(): Promise<Settings>;
    setSettings(settings: Settings): Promise<{
        success: boolean;
    }>;
}
//# sourceMappingURL=MetadataManager.d.ts.map