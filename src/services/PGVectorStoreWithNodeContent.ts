import { PGVectorStore } from '@llamaindex/postgres';
import { nodeToMetadata, metadataDictToNode } from '@llamaindex/core/vector-store';
import { MetadataMode, type BaseNode } from '@llamaindex/core/schema';
import type { VectorStoreQuery, VectorStoreQueryResult } from '@llamaindex/core/vector-store';

// @llamaindex/postgres (archived upstream) stores only the flat node.metadata and rebuilds results as plain
// Documents, losing the node's type, relationships, and excluded-metadata keys. This subclass stores the
// serialized node in the row's metadata (`_node_content`, as LlamaIndex Python does) and restores it on query.
// Rows written without `_node_content` (by the stock store) are returned as the stock store returns them.
// Implements https://github.com/run-llama/LlamaIndexTS/pull/2232 (never merged; the repo is archived), which fixes
// https://github.com/run-llama/LlamaIndexTS/issues/2225. Replaces the old postgres_index.js.patch.
export class PGVectorStoreWithNodeContent extends PGVectorStore {
  constructor(config: ConstructorParameters<typeof PGVectorStore>[0]) {
    super(config);
    // getDataToInsert is private in the base class's types, so it can't be overridden as a method
    const baseGetDataToInsert = (this as any).getDataToInsert.bind(this);
    (this as any).getDataToInsert = (embeddingResults: BaseNode[]) => {
      // before the base class runs: it stamps a Date onto node.metadata, which nodeToMetadata would reject as non-flat
      const metas = embeddingResults.map((node) => {
        const meta: Record<string, any> = nodeToMetadata(node, true, 'text', true);
        meta.create_date ??= new Date();
        return meta;
      });
      return (baseGetDataToInsert(embeddingResults) as any[][]).map((row, i) => [row[0], row[1], row[2], row[3], metas[i], row[5]]);
    };
  }

  async query(query: VectorStoreQuery, options?: object): Promise<VectorStoreQueryResult> {
    const result = await super.query(query, options);
    result.nodes = result.nodes?.map((doc) => {
      if (!doc.metadata?._node_content) return doc;
      const node = metadataDictToNode(doc.metadata);
      node.setContent(doc.getContent(MetadataMode.NONE)); // the row's `document` column holds the text; _node_content has it blanked
      node.embedding = doc.embedding; // _node_content is stored without it; the embeddings column has it
      return node;
    });
    return result;
  }
}
