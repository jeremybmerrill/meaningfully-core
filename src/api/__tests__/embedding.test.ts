//@ts-nocheck
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createEmbeddings, previewResults, previewSample, getDocStore, getIndex, search } from '../embedding.js';
import { loadDocumentsFromCsv } from '../../services/csvLoader.js';
import { transformDocumentsToNodes, estimateCost, searchDocuments, getExistingVectorStoreIndex, persistNodes, getStorageContext } from '../../services/embeddings.js';
import { MetadataMode } from 'llamaindex';

// filepath: /Users/jeremybmerrill/code/meaningfully/src/main/api/embedding.test.ts


vi.mock('../../services/csvLoader');
vi.mock('../../services/embeddings');

describe('embedding.ts', () => {
    describe('createEmbeddings', () => {
        // createEmbeddings takes documents directly (loading the CSV and checking for an
        // empty result are the caller's job, e.g. Meaningfully.uploadCsv) -- there's no
        // "empty documents" case to test here since the caller never reaches this function
        // with an empty array.
        it('should create embeddings and return success', async () => {
            const mockDocuments = [{ text: 'doc1', metadata: {} }, { text: 'doc2', metadata: {} }];
            const mockNodes = [{ text: 'node1', metadata: {} }, { text: 'node2', metadata: {} }];
            const mockIndex = 'index1';
            transformDocumentsToNodes.mockResolvedValue(mockNodes);
            persistNodes.mockResolvedValue(mockIndex);

            const result = await createEmbeddings(mockDocuments, {}, {}, {});

            expect(result).toEqual({ success: true, index: mockIndex });
        });

        it('should return error on failure', async () => {
            transformDocumentsToNodes.mockRejectedValue(new Error('Failed to transform documents'));

            const result = await createEmbeddings([{ text: 'doc1', metadata: {} }], {}, {}, {});

            expect(result).toEqual({ success: false, error: 'Failed to transform documents' });
        });
    });

    describe('previewResults', () => {
        it('should return preview results, estimated cost, and a sample for later reuse', async () => {
            const mockDocuments = Array(20).fill(null).map((_, i) => ({ text: `doc${i}`, metadata: { row: i } }));
            const mockNodes = [{ text: 'node1', metadata: {} }, { text: 'node2', metadata: {} }];
            const mockEstimate = { estimatedPrice: 10, tokenCount: 100, pricePer1M: 0.01 };
            transformDocumentsToNodes.mockResolvedValue(mockNodes);
            estimateCost.mockReturnValue(mockEstimate);

            const result = await previewResults(mockDocuments, {});

            expect(result).toEqual({
                success: true,
                nodes: mockNodes,
                ...mockEstimate,
                documentCount: 20,
                sample: mockDocuments.slice(10, 20).map((d) => ({ text: d.text, metadata: d.metadata })),
            });
        });

        it('should return error on failure', async () => {
            transformDocumentsToNodes.mockRejectedValue(new Error('Failed to transform documents'));

            const result = await previewResults([{ text: 'doc', metadata: {} }], {});

            expect(result).toEqual({ success: false, error: 'Failed to transform documents' });
        });
    });

    describe('previewSample', () => {
        it('extrapolates estimated cost from the sample to the full document count', async () => {
            const mockSample = [{ text: 'doc1', metadata: {} }, { text: 'doc2', metadata: {} }];
            const mockNodes = [{ text: 'node1', metadata: {} }];
            transformDocumentsToNodes.mockResolvedValue(mockNodes);
            estimateCost.mockReturnValue({ estimatedPrice: 10, tokenCount: 100, pricePer1M: 0.01 });

            // sample of 2 representing 20 total documents -> 10x scale
            const result = await previewSample(mockSample, 20, {});

            expect(result).toEqual({
                success: true,
                nodes: mockNodes,
                estimatedPrice: 100,
                tokenCount: 1000,
                pricePer1M: 0.01,
            });
        });

        it('returns an error when there is no sample to work from', async () => {
            const result = await previewSample([], 20, {});

            expect(result).toEqual({ success: false, error: 'No sample data available for preview.' });
        });

        it('returns error on failure', async () => {
            transformDocumentsToNodes.mockRejectedValue(new Error('Failed to transform documents'));

            const result = await previewSample([{ text: 'doc', metadata: {} }], 20, {});

            expect(result).toEqual({ success: false, error: 'Failed to transform documents' });
        });
    });

    describe('getDocStore', () => {
        it('should return existing doc store', async () => {
            const mockDocStore = 'docStore';
            getStorageContext.mockResolvedValue({ docStore: mockDocStore });

            const result = await getDocStore({}, {}, {});

            expect(result).toBe(mockDocStore);
        });
    });

    describe('getIndex', () => {
        it('should return existing vector store index', async () => {
            const mockIndex = 'index';
            getExistingVectorStoreIndex.mockResolvedValue(mockIndex);

            const result = await getIndex({}, {});

            expect(result).toBe(mockIndex);
        });
    });

    describe('search', () => {
        it('should return search results', async () => {
            const mockResults = [
                { node: { getContent: () => 'content1', metadata: {} }, score: 1 },
                { node: { getContent: () => 'content2', metadata: {} }, score: 2 }
            ];
            searchDocuments.mockResolvedValue({ results: mockResults, hasMore: false });

            const result = await search('index', 'query');

            expect(result).toEqual({
                results: [
                    { text: 'content1', score: 1, metadata: {} },
                    { text: 'content2', score: 2, metadata: {} }
                ],
                hasMore: false
            });
        });

        it('should handle no search results', async () => {
            searchDocuments.mockResolvedValue({ results: [], hasMore: false });

            const result = await search('index', 'query');

            expect(result).toEqual({ results: [], hasMore: false });
        });

        it('should handle search results with null scores', async () => {
            const mockResults = [
                { node: { getContent: () => 'content1', metadata: {} }, score: null },
                { node: { getContent: () => 'content2', metadata: {} }, score: null }
            ];
            searchDocuments.mockResolvedValue({ results: mockResults, hasMore: false });

            const result = await search('index', 'query');

            expect(result).toEqual({
                results: [
                    { text: 'content1', score: 0, metadata: {} },
                    { text: 'content2', score: 0, metadata: {} }
                ],
                hasMore: false
            });
        });
    });
});

describe('createEmbeddings with progress tracking', () => {
      beforeEach(() => {
        vi.clearAllMocks();
      });

      it('should pass progress callback to persistNodes', async () => {
        // Setup mocks
        const mockDocuments = [{ text: 'doc1' }, { text: 'doc2' }];
        const mockNodes = [{ text: 'node1', metadata: {} }, { text: 'node2', metadata: {} }];
        const mockIndex = 'index1';
        loadDocumentsFromCsv.mockResolvedValue(mockDocuments);
        transformDocumentsToNodes.mockResolvedValue(mockNodes);
        persistNodes.mockImplementation((nodes, config, settings, clients, callback) => {
          // Call the callback with sample progress
          if (callback) {
            callback(1, 2);  // 50% progress
            callback(2, 2);  // 100% progress
          }
          return Promise.resolve(mockIndex);
        });

        // Execute
        const result = await createEmbeddings('path/to/csv', 'text', { modelName: 'test-model' }, {}, {});

        // Verify
        expect(persistNodes).toHaveBeenCalledTimes(1);
        expect(persistNodes.mock.calls[0][4]).toBeInstanceOf(Function); // Verify callback was passed
        expect(result).toEqual({ success: true, index: mockIndex });
      });

      it('should correctly track progress through ProgressManager', async () => {
        // Setup
        vi.mock('../../services/progressManager', () => {
          const mockInstance = {
            startOperation: vi.fn(),
            updateProgress: vi.fn(),
            completeOperation: vi.fn(),
            clearOperation: vi.fn()
          };
          
          return {
            ProgressManager: {
              getInstance: () => mockInstance
            }
          };
        });
        
        // Re-import to use mocked version
        const { createEmbeddings } = await import('../embedding.js');
        const { ProgressManager } = await import('../../services/progressManager.js');
        
        const mockDocuments = [{ text: 'doc1' }, { text: 'doc2' }];
        const mockNodes = [{ text: 'node1', metadata: {} }];
        const mockIndex = 'testIndex';
        
        loadDocumentsFromCsv.mockResolvedValue(mockDocuments);
        transformDocumentsToNodes.mockResolvedValue(mockNodes);
        persistNodes.mockImplementation((nodes, config, settings, clients, callback) => {
          if (callback) callback(1, 2); // Call with 50% progress
          return Promise.resolve(mockIndex);
        });
        
        // Execute
        await createEmbeddings('path/to/csv', 'text', {}, {}, {});
        
        // Verify
        const progressManager = ProgressManager.getInstance();
        expect(progressManager.startOperation).toHaveBeenCalledWith(expect.stringMatching(/^embed-\d+$/), 100);
        expect(progressManager.updateProgress).toHaveBeenCalledWith(expect.any(String), 5);
        expect(progressManager.updateProgress).toHaveBeenCalledWith(expect.any(String), expect.any(Number));
        expect(progressManager.completeOperation).toHaveBeenCalledWith(expect.any(String));
      });

      it('should properly calculate percentage in progress callback', async () => {
        // Setup mocks with spy on updateProgress
        vi.mock('../../services/progressManager', () => {
          const mockInstance = {
            startOperation: vi.fn(),
            updateProgress: vi.fn(),
            completeOperation: vi.fn(),
            clearOperation: vi.fn()
          };
          
          return {
            ProgressManager: {
              getInstance: () => mockInstance
            }
          };
        });
        
        // Re-import to use mocked version
        const { createEmbeddings } = await import('../embedding.js');
        const { ProgressManager } = await import('../../services/progressManager.js');

        const mockDocuments = [{ text: 'doc1' }];
        const mockNodes = [{ text: 'node1', metadata: {} }];
        loadDocumentsFromCsv.mockResolvedValue(mockDocuments);
        transformDocumentsToNodes.mockResolvedValue(mockNodes);
        
        // Simulate persistNodes calling the callback with various progress values
        persistNodes.mockImplementation((nodes, config, settings, clients, callback) => {
          if (callback) {
            callback(0, 10);  // 0% progress
            callback(5, 10);  // 50% progress
            callback(10, 10); // 100% progress
          }
          return Promise.resolve('mockIndex');
        });
        
        // Execute
        await createEmbeddings('path/to/csv', 'text', {}, {}, {});
        
        // Verify percentage calculations
        // Initial update at 5%
        const progressManager = ProgressManager.getInstance();
        expect(progressManager.updateProgress).toHaveBeenCalledWith(expect.any(String), 5);
        
        // Progress updates: 0%, 50%, 100% mapped to 5-95% range
        expect(progressManager.updateProgress).toHaveBeenCalledWith(expect.any(String), 5);   // 0% -> 5%
        expect(progressManager.updateProgress).toHaveBeenCalledWith(expect.any(String), 50);  // 50% -> 50%
        expect(progressManager.updateProgress).toHaveBeenCalledWith(expect.any(String), 95);  // 100% -> 95%
      });

      // Note: there's no "empty documents" case to test here -- createEmbeddings takes
      // documents directly and never checks for emptiness itself (the caller does, before
      // ever calling this function), so progress tracking never starts in that case either.

      it('shoulde complete operation on successful embedding', async () => {
        // Setup
        vi.mock('../../services/progressManager', () => {
          const mockInstance = {
            startOperation: vi.fn(),
            updateProgress: vi.fn(),
            completeOperation: vi.fn(),
            clearOperation: vi.fn()
          };
          
          return {
            ProgressManager: {
              getInstance: () => mockInstance
            }
          };
        });
        
        // Re-import to use mocked version
        const { createEmbeddings } = await import('../embedding.js');
        const { ProgressManager } = await import('../../services/progressManager.js');

        const mockDocuments = [{ text: 'doc1' }];
        const mockNodes = [{ text: 'node1', metadata: {} }];
        loadDocumentsFromCsv.mockResolvedValue(mockDocuments);
        transformDocumentsToNodes.mockResolvedValue(mockNodes);
        persistNodes.mockResolvedValue('mockIndex');
        
        // Execute
        await createEmbeddings('path/to/csv', 'text', {}, {}, {});
        
        // Verify
        const progressManager = ProgressManager.getInstance();
        expect(progressManager.completeOperation).toHaveBeenCalled();
      });
    });