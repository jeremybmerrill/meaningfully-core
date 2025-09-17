import { MetadataManager } from '../MetadataManager.js';
import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('MetadataManager', () => {
  let metadataManager: MetadataManager;

  beforeEach(() => {
    metadataManager = new (class extends MetadataManager {
      protected async runQuery<T>(query: string, params?: any[]): Promise<T[]> {
        return [] as T[];
      }
      protected async runQuerySingle<T>(query: string, params?: any[]): Promise<T | null> {
        return null;
      }
      protected async initializeDatabase(): Promise<void> {}
      protected close(): void {}
    })();
  });

  it('should add a document set and return its ID', async () => {
    vi.spyOn(metadataManager, 'runQuerySingle').mockResolvedValueOnce({ set_id: 1 });

    const documentSetId = await metadataManager.addDocumentSet({
      name: 'Test Set',
      uploadDate: new Date(),
      parameters: {},
      totalDocuments: 10,
    });

    expect(documentSetId).toBe(1);
  });

  it('should retrieve a document set by ID', async () => {
    vi.spyOn(metadataManager, 'runQuerySingle').mockResolvedValueOnce({
      set_id: 1,
      name: 'Test Set',
      upload_date: new Date().toISOString(),
      parameters: '{}',
      total_documents: 10,
    });

    const documentSet = await metadataManager.getDocumentSet(1);

    expect(documentSet).toEqual({
      documentSetId: 1,
      name: 'Test Set',
      uploadDate: expect.any(Date),
      parameters: {},
      totalDocuments: 10,
    });
  });

  it('should update the document count for a document set', async () => {
    const runQuerySpy = vi.spyOn(metadataManager, 'runQuery').mockResolvedValueOnce([]);

    await metadataManager.updateDocumentCount(1, 5);

    expect(runQuerySpy).toHaveBeenCalledWith(metadataManager.queries.updateDocumentCount, [5, 1]);
  });

  it('should delete a document set by ID', async () => {
    const runQuerySpy = vi.spyOn(metadataManager, 'runQuery').mockResolvedValueOnce([]);

    await metadataManager.deleteDocumentSet(1);

    expect(runQuerySpy).toHaveBeenCalledWith(metadataManager.queries.deleteDocumentSet, [1]);
  });

  it('should retrieve default settings if none exist', async () => {
    vi.spyOn(metadataManager, 'runQuerySingle').mockResolvedValueOnce(null);

    const settings = await metadataManager.getSettings();

    expect(settings).toEqual({
      openAIKey: null,
      oLlamaBaseURL: null,
      azureOpenAIKey: null,
      azureOpenAIEndpoint: null,
      azureOpenAIApiVersion: '2024-02-01',
      mistralApiKey: null,
      geminiApiKey: null,
    });
  });

  it('should update settings', async () => {
    const runQuerySpy = vi.spyOn(metadataManager, 'runQuery').mockResolvedValueOnce([]);

    const result = await metadataManager.setSettings({
      openAIKey: 'test-key',
      oLlamaBaseURL: 'http://localhost',
      azureOpenAIKey: 'azure-key',
      azureOpenAIEndpoint: 'http://azure.endpoint',
      azureOpenAIApiVersion: '2024-02-01',
      mistralApiKey: 'mistral-key',
      geminiApiKey: 'gemini-key',
    });

    expect(result).toEqual({ success: true });
    expect(runQuerySpy).toHaveBeenCalledWith(metadataManager.queries.upsertSettings, [
      JSON.stringify({
        openAIKey: 'test-key',
        oLlamaBaseURL: 'http://localhost',
        azureOpenAIKey: 'azure-key',
        azureOpenAIEndpoint: 'http://azure.endpoint',
        azureOpenAIApiVersion: '2024-02-01',
        mistralApiKey: 'mistral-key',
        geminiApiKey: 'gemini-key',
      }),
      JSON.stringify({
        openAIKey: 'test-key',
        oLlamaBaseURL: 'http://localhost',
        azureOpenAIKey: 'azure-key',
        azureOpenAIEndpoint: 'http://azure.endpoint',
        azureOpenAIApiVersion: '2024-02-01',
        mistralApiKey: 'mistral-key',
        geminiApiKey: 'gemini-key',
      }),
    ]);
  });
});