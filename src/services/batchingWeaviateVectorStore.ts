import { WeaviateVectorStore } from '@llamaindex/weaviate';
import { BaseNode, type VectorStoreQuery, type VectorStoreQueryResult, metadataDictToNode } from 'llamaindex';
import {
  type FilterValue,
} from "weaviate-client";
/*

Patched version of WeaviateVectorStore to handle large batches by splitting into smaller chunks.

When I loaded a large-ish (5.4MB) spreadsheet, I got a Weaviate error about trying to load too much data at once.

*/

export class BatchingWeaviateVectorStore extends WeaviateVectorStore {
  async add(nodes: BaseNode[]): Promise<string[]> {
    const batchSize = 100; // Define the batch size
    const results: string[] = []; // Collect results from each batch
    for (let i = 0; i < nodes.length; i += batchSize) {
      const batch = nodes.slice(i, i + batchSize);
      const batchResults = await super.add(batch); // Call the parent class's add method for each batch
      results.push(...batchResults); // Aggregate results
    }
    return results; // Return aggregated results
  }


  // public async query(
  //   query: VectorStoreQuery & {
  //     queryStr: string;
  //   },
  // ): Promise<VectorStoreQueryResult> {
  //   const collection = await this.ensureCollection();
  //   const allProperties = await this.getAllProperties();

  //   let filters: FilterValue | undefined = undefined;

  //   if (query.docIds) {
  //     filters = collection.filter
  //       .byProperty("doc_id")
  //       .containsAny(query.docIds);
  //   }

  //   if (query.filters) {
  //     filters = toWeaviateFilter(collection, query.filters);
  //   }

  //   const hybridOptions: BaseHybridOptions<undefined> = {
  //     returnMetadata: Object.values(SIMILARITY_KEYS),
  //     returnProperties: allProperties,
  //     includeVector: true,
  //   };
  //   const alpha = this.getQueryAlpha(query);
  //   if (query.queryEmbedding) {
  //     hybridOptions.vector = query.queryEmbedding;
  //   }
  //   if (query.similarityTopK) {
  //     hybridOptions.limit = query.similarityTopK;
  //   }
  //   if (alpha) {
  //     hybridOptions.alpha = alpha;
  //   }
  //   if (filters) {
  //     hybridOptions.filters = filters;
  //   }

  //   const queryResult = await collection.query.hybrid(
  //     query.queryStr,
  //     hybridOptions,
  //   );

  //   const entries = queryResult.objects;

  //   const similarityKey = SIMILARITY_KEYS[query.mode];
  //   const nodes: BaseNode<Metadata>[] = [];
  //   const similarities: number[] = [];
  //   const ids: string[] = [];

  //   entries.forEach((entry, index) => {
  //     if (index < query.similarityTopK && entry.metadata) {
  //       const node = metadataDictToNode(entry.properties);
  //       node.setContent(entry.properties[this.contentKey]);
  //       nodes.push(node);
  //       ids.push(entry.uuid);
  //       similarities.push(this.getNodeSimilarity(entry, similarityKey));
  //     }
  //   });

  //   return {
  //     nodes,
  //     similarities,
  //     ids,
  //   };
  // }  
}



