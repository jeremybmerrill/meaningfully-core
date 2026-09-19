//@ts-nocheck

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Document, TextNode } from 'llamaindex';

const { bm25RetrieveMock, Bm25RetrieverMock } = vi.hoisted(() => {
  const bm25RetrieveMock = vi.fn();
  const Bm25RetrieverMock = vi.fn().mockImplementation((options) => ({
    options,
    retrieve: bm25RetrieveMock
  }));
  return { bm25RetrieveMock, Bm25RetrieverMock };
});
vi.mock('../bm25Retriever.js', () => ({ Bm25Retriever: Bm25RetrieverMock }));

// First, set up the mock before importing the module
vi.mock(import("../embeddings.js"), async (importOriginal) => {
  const actual = await importOriginal()
  return {
    ...actual,
    // your mocked methods
    estimateCost: vi.fn(),
    getExistingVectorStoreIndex: vi.fn(),
    persistNodes: vi.fn(),
    persistDocuments: vi.fn(),
    getExistingDocStore: vi.fn(),
    searchDocuments: vi.fn()
  }
})

// Now import the mocked functions
import { transformDocumentsToNodes, getEmbedModel, getOllamaEmbeddingModels, getLMStudioEmbeddingModels, getEmbeddingDimensions, searchDocumentsHybrid } from '../embeddings.js';
import { LMStudioEmbedding } from '../lmStudioEmbedding.js';

function nodeWithScore(id: string, score: number) {
  return { node: { id_: id, getContent: () => id, metadata: {} }, score };
}

describe('transformDocumentsToNodes', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockConfig = {
    chunkSize: 100,
    chunkOverlap: 10,
    combineSentencesIntoChunks: true,
    sploderMaxSize: 500,
    modelProvider: 'mock',
    modelName: 'text-embedding-3-small',
    vectorStoreType: "simple" as "simple",
    storagePath: './storage',
    projectName: 'test_project',
    splitIntoSentences: true,
  };

  const mockSettings = {
    openAIKey: 'mock-api-key',
    oLlamaBaseURL: 'http://localhost',
    lmStudioBaseURL: 'http://localhost:1234',
    azureOpenAIKey: null,
    azureOpenAIEndpoint: null,
    azureOpenAIApiVersion: null,
    mistralApiKey: null,
    geminiApiKey: null,
  };

  it('should process documents and return nodes', async () => {
    const mockDocuments = [
      new Document({ text: 'Document 1', metadata: { key1: 'value1' } }),
      new Document({ text: 'Document 2', metadata: { key2: 'value2' } }),
    ];
    const mockNodes = [
      new TextNode({ text: 'Document 1' }),
      new TextNode({ text: 'Document 2' }),
    ];

    const result = await transformDocumentsToNodes(mockDocuments, mockConfig, mockSettings);

    expect(result.map((node) => node.text)).toEqual(mockNodes.map((node) => node.text));
  });

  it('should filter out documents with null, undefined, or zero-length text', async () => {
    const mockDocuments = [
      new Document({ text: 'Valid Document', metadata: { key1: 'value1' } }),
      new Document({ text: undefined, metadata: { key3: 'value3' } }),
      new Document({ text: '', metadata: { key4: 'value4' } }),
    ];
    const filteredDocuments = [mockDocuments[0]];
    const mockNodes = [new TextNode({ text: 'Valid Document' })];

    // (transformDocumentsToNodes as vi.Mock).mockResolvedValue(mockNodes);

    const result = await transformDocumentsToNodes(mockDocuments, mockConfig, mockSettings);
    expect(result.map((n) => n.text)).toEqual(mockNodes.map((n) => n.text));
    
    // TODO: I can't get these to work. Apparently you can't spyOn a function that is imported from the same file.
    // all well and good but ... why did CoPilot generate a test that can't work?
    // expect(transformDocumentsToNodes).toHaveBeenCalledWith(filteredDocuments, expect.any(Array));
  });

  it('should exclude all metadata keys from embedding', async () => {
    const mockDocuments = [
      new Document({ text: 'Document 1', metadata: { key1: 'value1', key2: 'value2' } }),
    ];

    const nodes = await transformDocumentsToNodes(mockDocuments, mockConfig, mockSettings)
    expect(nodes[0].excludedEmbedMetadataKeys).toEqual(['key1', 'key2']);
  });
});

describe('getEmbedModel', () => {
  const mockConfig = {
    chunkSize: 100,
    chunkOverlap: 10,
    combineSentencesIntoChunks: true,
    sploderMaxSize: 500,
    modelProvider: 'openai',
    modelName: 'text-embedding-3-small',
    vectorStoreType: "simple" as "simple",
    storagePath: './storage',
    projectName: 'test_project',
    splitIntoSentences: true,
  };

  const mockSettings = {
    openAIKey: 'mock-api-key',
    oLlamaBaseURL: 'http://localhost',
    lmStudioBaseURL: 'http://localhost:1234',
    azureOpenAIKey: null,
    azureOpenAIEndpoint: null,
    azureOpenAIApiVersion: null,
    mistralApiKey: null,
    geminiApiKey: null,
  };


  it('should handle different model providers correctly', () => {
    // Test with 'ollama' provider
    const ollamaModel = getEmbedModel(
      { ...mockConfig, modelProvider: 'ollama' }, 
      mockSettings
    );
    expect(ollamaModel).toBeDefined();
    
    // Test with 'lmstudio' provider
    const lmStudioModel = getEmbedModel(
      { ...mockConfig, modelProvider: 'lmstudio' },
      mockSettings
    );
    expect(lmStudioModel).toBeInstanceOf(LMStudioEmbedding);

    // Test with 'mock' provider
    const mockModel = getEmbedModel(
      { ...mockConfig, modelProvider: 'mock' },
      mockSettings
    );
    expect(mockModel).toBeDefined();

    // Test with invalid provider
    expect(() => {
      getEmbedModel(
        { ...mockConfig, modelProvider: 'invalid' as any },
        mockSettings
      );
    }).toThrow('Unsupported embedding model provider: invalid');
  });

  it('requires a base URL for the lmstudio provider', () => {
    expect(() => {
      getEmbedModel(
        { ...mockConfig, modelProvider: 'lmstudio' },
        { ...mockSettings, lmStudioBaseURL: null }
      );
    }).toThrow('LM Studio base URL is required for LM Studio embedding models');
  });
});

describe('getOllamaEmbeddingModels', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  it('returns only models whose capabilities include "embedding"', async () => {
    (fetch as any).mockImplementation((url: string, opts?: any) => {
      if (url.endsWith('/api/tags')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            models: [{ name: 'mxbai-embed-large:latest' }, { name: 'llama3.2:latest' }],
          }),
        });
      }
      if (url.endsWith('/api/show')) {
        const { model } = JSON.parse(opts.body);
        const capabilities = model === 'mxbai-embed-large:latest' ? ['embedding'] : ['completion'];
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ capabilities }) });
      }
      throw new Error(`unexpected url ${url}`);
    });

    const models = await getOllamaEmbeddingModels('http://localhost:11434/');
    expect(models).toEqual(['mxbai-embed-large:latest']);
  });

  it('throws when the Ollama server is unreachable or returns an error', async () => {
    (fetch as any).mockResolvedValue({ ok: false, status: 500, statusText: 'Internal Server Error' });

    await expect(getOllamaEmbeddingModels('http://localhost:11434')).rejects.toThrow('Failed to list Ollama models');
  });
});

describe('getLMStudioEmbeddingModels', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  it('returns only models whose type is "embeddings"', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.endsWith('/api/v0/models')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            data: [
              { id: 'meta-llama-3.1-8b-instruct', object: 'model', type: 'llm' },
              { id: 'text-embedding-nomic-embed-text-v1.5', object: 'model', type: 'embeddings' },
            ],
          }),
        });
      }
      throw new Error(`unexpected url ${url}`);
    });

    const models = await getLMStudioEmbeddingModels('http://localhost:1234/');
    expect(models).toEqual(['text-embedding-nomic-embed-text-v1.5']);
  });

  it('throws when the LM Studio server is unreachable or returns an error', async () => {
    (fetch as any).mockResolvedValue({ ok: false, status: 500, statusText: 'Internal Server Error' });

    await expect(getLMStudioEmbeddingModels('http://localhost:1234')).rejects.toThrow('Failed to list LM Studio models');
  });
});

describe('getEmbeddingDimensions', () => {
  it('returns the known dimension without making an embedding call for a well-known model', async () => {
    const getTextEmbedding = vi.fn();
    const fakeEmbedModel = { getTextEmbedding } as any;

    const dimensions = await getEmbeddingDimensions(fakeEmbedModel, 'text-embedding-3-small');

    expect(dimensions).toBe(1536);
    expect(getTextEmbedding).not.toHaveBeenCalled();
  });

  it('determines the dimension by embedding a probe string for an unlisted model (e.g. an arbitrary Ollama/LM Studio model)', async () => {
    const getTextEmbedding = vi.fn().mockResolvedValue(new Array(768).fill(0));
    const fakeEmbedModel = { getTextEmbedding } as any;

    const dimensions = await getEmbeddingDimensions(fakeEmbedModel, 'some-arbitrary-local-model');

    expect(dimensions).toBe(768);
    expect(getTextEmbedding).toHaveBeenCalledWith(expect.any(String));
  });
});

describe('searchDocumentsHybrid', () => {
  const mockDocStore = {} as any;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fuses vector and BM25 rankings by reciprocal rank, without metadata filters', async () => {
    const vectorResults = [
      nodeWithScore('a', 0.9),
      nodeWithScore('b', 0.8),
      nodeWithScore('c', 0.7)
    ];
    const vectorRetrieveMock = vi.fn().mockResolvedValue(vectorResults);
    bm25RetrieveMock.mockResolvedValue([nodeWithScore('c', 5), nodeWithScore('a', 3)]);
    const asRetriever = vi.fn().mockReturnValue({ retrieve: vectorRetrieveMock });
    const fakeIndex = { asRetriever } as any;

    const { results, hasMore } = await searchDocumentsHybrid(fakeIndex, mockDocStore, 'query', 10, undefined, 0);

    // The vector leg casts a wider net so the BM25 leg ranks matching vector-node IDs, not
    // original documents from the doc store.
    expect(asRetriever).toHaveBeenCalledWith({ similarityTopK: 200, filters: { filters: [] } });
    expect(Bm25RetrieverMock).toHaveBeenCalledWith({
      docStore: mockDocStore,
      topK: 11,
      nodes: vectorResults.map((result) => result.node),
      docIds: ['a', 'b', 'c']
    });

    // 'a' ranks #1 in vector and #2 in BM25; 'c' ranks #3 in vector and #1 in BM25 -- 'a' wins
    // narrowly by finishing higher in more places, 'b' (BM25-absent) trails both.
    expect(results.map((r: any) => r.node.id_)).toEqual(['a', 'c', 'b']);
    expect(hasMore).toBe(false);

    // Displayed scores are the underlying vector similarity, not the fused RRF rank score, so
    // they aren't necessarily monotonically decreasing (here 'c' outranks 'b' despite a lower
    // similarity score, since 'c' also matched BM25).
    expect(results.map((r: any) => r.score)).toEqual([0.9, 0.7, 0.8]);
  });

  it('ignores BM25 results whose IDs do not belong to vector candidates', async () => {
    const vectorRetrieveMock = vi.fn().mockResolvedValue([nodeWithScore('a', 0.9)]);
    bm25RetrieveMock.mockResolvedValue([nodeWithScore('z', 5), nodeWithScore('a', 3)]);
    const asRetriever = vi.fn().mockReturnValue({ retrieve: vectorRetrieveMock });
    const fakeIndex = { asRetriever } as any;

    const { results } = await searchDocumentsHybrid(fakeIndex, mockDocStore, 'query', 10, undefined, 0);

    const scoresById = Object.fromEntries(results.map((r: any) => [r.node.id_, r.score]));
    expect(scoresById).toEqual({ a: 0.9 });
  });

  it('restricts the BM25 leg to the filtered vector candidate set via docIds, since Bm25Retriever cannot apply metadata filters directly', async () => {
    const vectorResults = [
      nodeWithScore('a', 0.9),
      nodeWithScore('b', 0.8),
      nodeWithScore('c', 0.7),
      nodeWithScore('d', 0.6)
    ];
    const vectorRetrieveMock = vi.fn().mockResolvedValue(vectorResults);
    bm25RetrieveMock.mockResolvedValue([nodeWithScore('d', 5), nodeWithScore('a', 3)]);
    const asRetriever = vi.fn().mockReturnValue({ retrieve: vectorRetrieveMock });
    const fakeIndex = { asRetriever } as any;
    const filters = [{ key: 'foo', operator: '==' as const, value: 'bar' }];

    const { results, hasMore } = await searchDocumentsHybrid(fakeIndex, mockDocStore, 'query', 2, filters, 0);

    // The vector leg casts a much wider net than requested (topK=2) so the filtered candidate
    // set handed to BM25 is close to complete, not just the first couple of matches.
    expect(asRetriever).toHaveBeenCalledWith({ similarityTopK: 200, filters: { filters } });
    expect(Bm25RetrieverMock).toHaveBeenCalledWith({
      docStore: mockDocStore,
      topK: 3,
      nodes: vectorResults.map((result) => result.node),
      docIds: ['a', 'b', 'c', 'd']
    });

    // Only the top 3 (retrievalDepth) of the 4 filtered vector results count toward fusion, so
    // 'd' -- 4th in the vector list -- is credited solely via its #1 BM25 ranking.
    expect(results.map((r: any) => r.node.id_)).toEqual(['a', 'd']);
    expect(hasMore).toBe(true);

    // 'd' still displays its real similarity score (0.6) despite ranking via BM25, since the
    // wider unfiltered vector leg had already computed it -- only a node with no similarity at
    // all (never seen by the vector leg) would fall back to 0.
    expect(results.map((r: any) => r.score)).toEqual([0.9, 0.6]);
  });
});