import { WeaviateVectorStore } from '@llamaindex/weaviate';
import { BaseNode } from 'llamaindex';

/*

Patched version of WeaviateVectorStore to handle large batches by splitting into smaller chunks.

When I loaded a large-ish (5.4MB) spreadsheet, I got a Weaviate error about trying to load too much data at once.

*/

// Weaviate rejects any property literally named "id" (reserved for the object's own UUID).
// A CSV column named "id" ends up as a metadata key of the same name, so we rename it going
// in and rename it back coming out, transparently to the rest of the app.
// ponytail: only "id" is escaped, since that's the only reported reserved-word collision.
const RESERVED_METADATA_KEY = 'id';
const ESCAPED_METADATA_KEY = '_csv_id';

function renameKey(obj: Record<string, any>, from: string, to: string): Record<string, any> {
  const { [from]: value, ...rest } = obj;
  return { ...rest, [to]: value };
}

export class BatchingWeaviateVectorStore extends WeaviateVectorStore {
  async add(nodes: BaseNode[]): Promise<string[]> {
    const batchSize = 100; // Define the batch size
    const results: string[] = []; // Collect results from each batch
    for (let i = 0; i < nodes.length; i += batchSize) {
      const batch = nodes.slice(i, i + batchSize).map((node) => {
        if (!node.metadata || !(RESERVED_METADATA_KEY in node.metadata)) return node;
        const escapedNode = node.clone();
        escapedNode.metadata = renameKey(escapedNode.metadata, RESERVED_METADATA_KEY, ESCAPED_METADATA_KEY);
        return escapedNode;
      });
      const batchResults = await super.add(batch); // Call the parent class's add method for each batch
      results.push(...batchResults); // Aggregate results
    }
    return results; // Return aggregated results
  }

  async query(query: any) {
    const result = await super.query(query);
    for (const node of result.nodes ?? []) {
      if (node.metadata && ESCAPED_METADATA_KEY in node.metadata) {
        node.metadata = renameKey(node.metadata, ESCAPED_METADATA_KEY, RESERVED_METADATA_KEY);
      }
    }
    return result;
  }
}
