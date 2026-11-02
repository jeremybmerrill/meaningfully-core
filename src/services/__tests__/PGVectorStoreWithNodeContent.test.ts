import { MockEmbedding } from '../mockEmbedding.js';
import { describe, it, expect } from 'vitest';
import { TextNode, Document, MetadataMode } from '@llamaindex/core/schema';
import { PGVectorStoreWithNodeContent } from '../PGVectorStoreWithNodeContent.js';

const makeStore = (rows: any[] = []) => {
  const store = new PGVectorStoreWithNodeContent({ client: {} as any, tableName: 't', dimensions: 2, embeddingModel: new MockEmbedding() });
  (store as any).getDb = async () => ({ query: async () => rows });
  return store;
};

describe('PGVectorStoreWithNodeContent', () => {
  it('stores the serialized node in metadata, and query restores it', async () => {
    const node = new TextNode({ id_: 'n1', text: 'hello', metadata: { speaker: 'a' }, embedding: [1, 2], excludedEmbedMetadataKeys: ['speaker'] });
    const [, , , document, meta] = (makeStore() as any).getDataToInsert([node])[0];
    expect(document).toBe('hello');
    expect(meta.speaker).toBe('a');
    expect(JSON.parse(meta._node_content).text).toBe(''); // text lives in the `document` column
    const [restored] = (await makeStore([{ id: 'n1', document, metadata: meta, embeddings: '[1,2]', s: 0.1 }]).query({ queryEmbedding: [1, 2], similarityTopK: 1 } as any)).nodes!;
    expect(restored!.getContent(MetadataMode.NONE)).toBe('hello');
    expect(restored!.embedding).toEqual([1, 2]);
    expect((restored as TextNode).excludedEmbedMetadataKeys).toEqual(['speaker']);
  });

  it('returns rows lacking _node_content as plain documents instead of throwing', async () => {
    const { nodes } = await makeStore([{ id: 'x', document: 'old', metadata: { a: 'b' }, embeddings: '[1,2]', s: 0 }]).query({ queryEmbedding: [1, 2], similarityTopK: 1 } as any);
    expect(nodes![0]).toBeInstanceOf(Document);
    expect(nodes![0]!.getContent(MetadataMode.NONE)).toBe('old');
  });
});
