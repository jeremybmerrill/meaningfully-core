import { MetadataManager } from '../MetadataManager.js';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { Knex } from 'knex';

describe('MetadataManager', () => {
  let metadataManager: MetadataManager;
  let mockKnex: any;

  beforeEach(() => {
    // Create a mock Knex instance with chainable methods
    mockKnex = {
      schema: {
        hasTable: vi.fn().mockResolvedValue(true),
        createTable: vi.fn().mockReturnThis(),
      },
      insert: vi.fn().mockReturnThis(),
      returning: vi.fn().mockResolvedValue([{ set_id: 1 }]),
      where: vi.fn().mockReturnThis(),
      first: vi.fn().mockResolvedValue(null),
      select: vi.fn().mockReturnThis(),
      orderBy: vi.fn().mockReturnThis(),
      limit: vi.fn().mockReturnThis(),
      offset: vi.fn().mockResolvedValue([]),
      count: vi.fn().mockReturnThis(),
      increment: vi.fn().mockResolvedValue(undefined),
      delete: vi.fn().mockResolvedValue(undefined),
      onConflict: vi.fn().mockReturnThis(),
      merge: vi.fn().mockResolvedValue(undefined),
      destroy: vi.fn().mockResolvedValue(undefined),
    };

    // Make mockKnex callable as a function that returns itself for chaining
    const knexFn = vi.fn().mockReturnValue(mockKnex);
    Object.assign(knexFn, mockKnex);

    metadataManager = new (class extends MetadataManager {
      protected knex = knexFn as any;
      protected async initializeDatabase(): Promise<void> {}
      protected close(): void {}
    })();
  });

  it('should add a document set and return its ID', async () => {
    mockKnex.returning.mockResolvedValueOnce([{ set_id: 1 }]);

    const documentSetId = await metadataManager.addDocumentSet({
      name: 'Test Set',
      uploadDate: new Date(),
      parameters: {},
      totalDocuments: 10,
    });

    expect(documentSetId).toBe(1);
  });

  it('should retrieve a document set by ID', async () => {
    mockKnex.first.mockResolvedValueOnce({
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
    await metadataManager.updateDocumentCount(1, 5);

    expect(mockKnex.increment).toHaveBeenCalledWith('total_documents', 5);
  });

  it('should delete a document set by ID', async () => {
    await metadataManager.deleteDocumentSet(1);

    expect(mockKnex.delete).toHaveBeenCalled();
  });

  it('should retrieve default settings if none exist', async () => {
    mockKnex.first.mockResolvedValueOnce(null);

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
    expect(mockKnex.insert).toHaveBeenCalled();
    expect(mockKnex.onConflict).toHaveBeenCalledWith('settings_id');
    expect(mockKnex.merge).toHaveBeenCalled();
  });
});