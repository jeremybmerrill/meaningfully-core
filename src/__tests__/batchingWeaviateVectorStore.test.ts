import { describe, it, expect, vi } from 'vitest';
import { Document } from 'llamaindex';

// Weaviate itself is never contacted in this test -- we stub out the base
// class's add/query so we can inspect exactly what BatchingWeaviateVectorStore
// sends and returns around it.
const mockAdd = vi.fn(async (nodes: any[]) => nodes.map((n) => n.id_));
const mockQuery = vi.fn(async () => ({ nodes: [] as any[], similarities: [], ids: [] }));

vi.mock('@llamaindex/weaviate', () => ({
  WeaviateVectorStore: class {
    constructor(_init: any) {}
    add(nodes: any[]) {
      return mockAdd(nodes);
    }
    query(query: any) {
      return mockQuery(query);
    }
  },
}));

const { BatchingWeaviateVectorStore } = await import('../services/batchingWeaviateVectorStore.js');

describe('BatchingWeaviateVectorStore', () => {
  it('escapes a metadata key named "id" before writing to Weaviate, without mutating the source node', async () => {
    const store = new (BatchingWeaviateVectorStore as any)({ weaviateClient: {} });
    const doc = new Document({ text: 'hello', metadata: { id: 'row-1', name: 'Alice' } });

    await store.add([doc]);

    const [passedNodes] = mockAdd.mock.calls[0];
    expect(passedNodes[0].metadata).not.toHaveProperty('id');
    expect(passedNodes[0].metadata._csv_id).toBe('row-1');
    expect(passedNodes[0].metadata.name).toBe('Alice');
    // the original node passed in by the caller must be untouched
    expect(doc.metadata.id).toBe('row-1');
  });

  it('leaves metadata untouched when there is no "id" key', async () => {
    const store = new (BatchingWeaviateVectorStore as any)({ weaviateClient: {} });
    const doc = new Document({ text: 'hello', metadata: { name: 'Bob' } });

    await store.add([doc]);

    const [passedNodes] = mockAdd.mock.calls.at(-1)!;
    expect(passedNodes[0]).toBe(doc);
    expect(passedNodes[0].metadata).toEqual({ name: 'Bob' });
  });

  it('unescapes the metadata key back to "id" after reading from Weaviate', async () => {
    const node = new Document({ text: 'hello', metadata: { _csv_id: 'row-1', name: 'Alice' } });
    mockQuery.mockResolvedValueOnce({ nodes: [node], similarities: [1], ids: ['uuid-1'] });

    const store = new (BatchingWeaviateVectorStore as any)({ weaviateClient: {} });
    const result = await store.query({} as any);

    expect(result.nodes[0].metadata.id).toBe('row-1');
    expect(result.nodes[0].metadata).not.toHaveProperty('_csv_id');
  });
});
