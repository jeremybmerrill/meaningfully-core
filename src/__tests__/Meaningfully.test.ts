import { describe, it, beforeEach, expect, vi } from 'vitest';
import { MetadataManager } from '../MetadataManager';
import fs from 'fs';
import path from 'path';
import { sanitizeProjectName } from '../utils.js';
import { createVectorStore, createDocumentStore, createIndexStore} from '../services/embeddings.js';
import { IndexStruct } from 'llamaindex';
import { Client } from 'pg'; // Import the real Postgres client


vi.mock('../MetadataManager');
vi.mock('fs');
vi.mock('path');

// Mock the embedding module before importing MeaningfullyAPI
vi.doMock('../api/embedding.js', () => ({
  getIndex: vi.fn(),
  search: vi.fn().mockResolvedValue({ results: [{ id: 1, text: 'result' }], hasMore: false }),
  createEmbeddings: vi.fn().mockResolvedValue({ success: true, error: null }),
}));
vi.doMock('../services/csvLoader.js', () => ({
  loadDocumentsFromCsv: vi.fn().mockResolvedValue([]),
}));
import { BaseNode } from 'llamaindex';

// Mock BaseNode so that getEmbeddings returns made up numbers
vi.mock('llamaindex', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    BaseNode: class extends actual.BaseNode {
      async getEmbeddings() {
        return [0.1, 0.2, 0.3, 0.4];
      }
      generateHash() {
        return 'hash';
      }
      getContent(){
        return "content";
      }
    }
  };
});

// Import MeaningfullyAPI after mocking
const { MeaningfullyAPI } = await import('../Meaningfully');

const FAKE_SETTINGS = {
        openAIKey: 'sk-proj-testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest',
        azureOpenAIKey: 'testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest',
        mistralApiKey: 'testtesttesttesttesttesttesttest',
        geminiApiKey:  'testtesttesttesttesttesttesttesttesttes',
        azureOpenAIApiVersion: "2024-02-01",
        azureOpenAIEndpoint: "https://test.openai.azure.com",
        oLlamaBaseURL: "http://localhost:11434",
      }

describe('MeaningfullyAPI', () => {
  let api: MeaningfullyAPI;
  let mockMetadataManager: MetadataManager;

  beforeEach(() => {
    // @ts-ignore
    mockMetadataManager = new MetadataManager() as MetadataManager;
    vi.spyOn(mockMetadataManager, 'addDocumentSet').mockResolvedValue(1);
    vi.spyOn(mockMetadataManager, 'getSettings').mockResolvedValue(FAKE_SETTINGS);
    vi.spyOn(mockMetadataManager, 'deleteDocumentSet').mockResolvedValue();
    api = new MeaningfullyAPI({
      storagePath: 'mock_storage_path',
      metadataManager: mockMetadataManager,
    });
  });

  describe('uploadCsv', () => {
    it('should upload a CSV and create embeddings successfully', async () => {
      const mockData = {
        filePath: '/mock/file.csv',
        datasetName: 'testDataset',
        textColumns: ['text'],
        metadataColumns: [],
        splitIntoSentences: true,
        combineSentencesIntoChunks: false,
        sploderMaxSize: 100,
        chunkSize: 512,
        chunkOverlap: 0,
        modelName: 'testModel',
        modelProvider: 'openai',
        description: 'Test dataset',
      };

      // Mock createEmbeddings for this test
      const createEmbeddingsMock = vi.spyOn(await import('../api/embedding.js'), 'createEmbeddings');
      createEmbeddingsMock.mockResolvedValue({ success: true });

      const result = await api.uploadCsv(mockData);

      expect(createEmbeddingsMock).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(String),
        expect.objectContaining({ modelName: 'testModel' }),
        expect.any(Object),
        expect.any(Object)
      );
      expect(result).toEqual({ success: true, documentSetId: 1 });

      createEmbeddingsMock.mockRestore(); // Restore the original implementation after the test
    });

    it('should handle errors during embeddings creation', async () => {
      const mockData = {
        filePath: '/mock/file.csv',
        datasetName: 'testDataset',
        textColumns: ['text'],
        metadataColumns: [],
        splitIntoSentences: true,
        combineSentencesIntoChunks: false,
        sploderMaxSize: 100,
        chunkSize: 512,
        chunkOverlap: 0,
        modelName: 'testModel',
        modelProvider: 'openai',
        description: 'Test dataset',
      };

      // Mock createEmbeddings to simulate an error
      const createEmbeddingsMock = vi.spyOn(await import('../api/embedding.js'), 'createEmbeddings');
      createEmbeddingsMock.mockResolvedValue({ success: false, error: 'Embedding error' });

      await expect(api.uploadCsv(mockData)).rejects.toThrow('Embedding error');
      expect(mockMetadataManager.deleteDocumentSet).toHaveBeenCalledWith(1);

      createEmbeddingsMock.mockRestore(); // Restore the original implementation after the test
    });
  });

  describe('searchDocumentSet', () => {
    it('should search a document set and return results', async () => {
      vi.spyOn(mockMetadataManager, 'getDocumentSet').mockResolvedValue({
        parameters: { modelName: 'testModel', modelProvider: 'openai', vectorStoreType: 'simple' },
        name: 'testDataset',
        documentSetId: 5,
        uploadDate: new Date(),
        totalDocuments: 420
      });

      const results = await api.searchDocumentSet(1, 'query', 10);

      expect(results).toEqual({ results: [{ id: 1, text: 'result' }], hasMore: false });
      expect(mockMetadataManager.getDocumentSet).toHaveBeenCalledWith(1);
    });
  });

  describe('deleteDocumentSet', () => {
    it('should delete a document set and associated files', async () => {
      vi.spyOn(mockMetadataManager, 'getDocumentSet').mockResolvedValue({
        parameters: { vectorStoreType: 'simple' },
        name: 'testDataset',
        documentSetId: 1,
        uploadDate: new Date(),
        totalDocuments: 100
      });

      vi.spyOn(fs, 'rmSync').mockImplementation(() => {});

      const result = await api.deleteDocumentSet(1);

      expect(mockMetadataManager.deleteDocumentSet).toHaveBeenCalledWith(1);
      expect(fs.rmSync).toHaveBeenCalledWith(
        path.join('mock_storage_path', 'testDataset'),
        { recursive: true, force: true }
      );
      expect(result).toEqual({ success: true });
    });
  });

  describe('getMaskedSettings', () => {
    it('should return masked settings', async () => {
      vi.spyOn(mockMetadataManager, 'getSettings').mockResolvedValue(FAKE_SETTINGS);

      const settings = await api.getMaskedSettings();

      expect(settings).toEqual({
        openAIKey: 'sk-proj-*******testtest',
        azureOpenAIKey: 'testtest*******testtest',
        mistralApiKey: 'testtest*******testtest',
        geminiApiKey: 'testtest*******ttesttes',
        azureOpenAIApiVersion: "2024-02-01",
        azureOpenAIEndpoint: "https://test.openai.azure.com",
        oLlamaBaseURL:  "http://localhost:11434",
      });
    });
  });
});


describe('MeaningfullyAPI - Store Deletion with Real Implementation', () => {
  let api: MeaningfullyAPI;
  let mockMetadataManager: MetadataManager;
  let realPostgresClient: Client;

  beforeEach(async () => {
    // @ts-ignore
    mockMetadataManager = new MetadataManager() as MetadataManager;

    // Initialize a real Postgres client
    realPostgresClient = new Client({
      connectionString: process.env.POSTGRES_CONNECTION_STRING,
    });
    await realPostgresClient.connect();

    api = new MeaningfullyAPI({
      storagePath: 'mock_storage_path',
      metadataManager: mockMetadataManager,
      postgresClient: realPostgresClient, // Use the real client
    });
    vi.unmock('fs')
    if (!fs.existsSync("mock_storage_path")){
      fs.mkdirSync("mock_storage_path");
    }
  });

  it('should create and delete a Postgres vector store using real implementation', async () => {
    const projectName = 'test_project';
    const sanitizedProjectName = sanitizeProjectName(projectName);
    const tableName = `vecs_${sanitizedProjectName}`;

    // Create the vector store using the real implementation
    const vectorStore = await createVectorStore(
      { vectorStoreType: 'postgres', projectName, storagePath: 'mock_storage_path', modelProvider: 'openai', modelName: 'text-embedding-ada-002' },
      FAKE_SETTINGS,
      api.getClients()
    );
    await vectorStore.add([new BaseNode({ id: '1', text: 'test document', embedding: Array(1536).fill(0.01) })]);

    // Verify the table exists
    const tableExistsQuery = `
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = $1
      );
    `;
    const tableExistsResult = await realPostgresClient.query(tableExistsQuery, [tableName]);
    expect(tableExistsResult.rows[0].exists).toBe(true);

    // Call the delete method
    await api.deletePostgresVectorStore(projectName);

    // Verify the table no longer exists
    const tableDeletedResult = await realPostgresClient.query(tableExistsQuery, [tableName]);
    expect(tableDeletedResult.rows[0].exists).toBe(false);
  });

  // it('should delete Simple vector store using real implementation', async () => {
  //   const projectName = 'test_project';
  //   const sanitizedProjectName = sanitizeProjectName(projectName);
  //   const storagePath = 'mock_storage_path';
  //   const persistDir = path.join(storagePath, sanitizedProjectName);

  //   // Create the vector store using the real implementation
  //   const vectorStore = await createVectorStore(
  //     { vectorStoreType: 'simple', projectName, storagePath, modelProvider: "openai", modelName: 'text-embedding-3-small' },
  //     FAKE_SETTINGS,
  //     api.getClients()
  //   );
  //   await vectorStore.add([new BaseNode({ id: '1', text: 'test document', embedding: [1,2,3] })]);
  //   await vectorStore.persist(path.join(persistDir, 'vector_store.json'));

  //   // Verify the vector store exists
  //   expect(fs.existsSync(path.join(persistDir, 'vector_store.json'))).toBe(true);

  //   // Call the delete method
  //   await api.deleteSimpleVectorStore(projectName);

  //   // Verify the vector store no longer exists
  //   expect(fs.existsSync(path.join(persistDir, 'vector_store.json'))).toBe(false);
  // });

  // it('should delete Simple document store using real implementation', async () => {
  //   const projectName = 'test_project';
  //   const sanitizedProjectName = sanitizeProjectName(projectName);
  //   const storagePath = 'mock_storage_path';
  //   const persistDir = path.join(storagePath, sanitizedProjectName);

  //   // Create the document store using the real implementation
  //   const docStore = await createDocumentStore(
  //     { vectorStoreType: 'simple', projectName, storagePath },
  //     FAKE_SETTINGS,
  //     api.getClients()
  //   );
  //   await docStore.addDocuments([new BaseNode({ id: '1', text: 'test document' })], true);
  //   await docStore.persist(path.join(persistDir, 'doc_store.json'));

  //   // Verify the document store exists
  //   expect(fs.existsSync(path.join(persistDir, 'doc_store.json'))).toBe(true);

  //   // Call the delete method
  //   await api.deleteSimpleDocStore(projectName);

  //   // Verify the document store no longer exists
  //   expect(fs.existsSync(path.join(persistDir, 'doc_store.json'))).toBe(false);
  // });

  // it('should delete Simple index store using real implementation', async () => {
  //   const projectName = 'test_project';
  //   const sanitizedProjectName = sanitizeProjectName(projectName);
  //   const storagePath = 'mock_storage_path';
  //   const persistDir = path.join(storagePath, sanitizedProjectName);

  //   // Create the index store using the real implementation
  //   const indexStore = await createIndexStore(
  //     { vectorStoreType: 'simple', projectName, storagePath },
  //     FAKE_SETTINGS,
  //     api.getClients()
  //   );
  //   indexStore.addIndexStruct(new IndexStruct({ summary: 'test document' }));
  //   await indexStore.persist(path.join(persistDir, 'index_store.json'));

  //   // Verify the index store exists
  //   expect(fs.existsSync(path.join(persistDir, 'index_store.json'))).toBe(true);

  //   // Call the delete method
  //   await api.deleteSimpleIndexStore(projectName);

  //   // Verify the index store no longer exists
  //   expect(fs.existsSync(path.join(persistDir, 'index_store.json'))).toBe(false);
  // });
});