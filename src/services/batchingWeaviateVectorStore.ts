import { WeaviateVectorStore } from '@llamaindex/weaviate';
import { BaseNode, type VectorStoreQuery, type VectorStoreQueryResult,   type MetadataFilter,
  type MetadataFilters, metadataDictToNode,   parseArrayValue, VectorStoreQueryMode,
  parseNumberValue, type Metadata } from 'llamaindex';
import {
  Filters,
  type FilterValue,
  type Collection,
} from "weaviate-client";
import type { BaseHybridOptions } from "weaviate-client";
/*

Patched version of WeaviateVectorStore to handle large batches by splitting into smaller chunks.

When I loaded a large-ish (5.4MB) spreadsheet, I got a Weaviate error about trying to load too much data at once.

*/


// copied verbatim from WeaviateVectorStore
const SIMILARITY_KEYS: {
  [key: string]: "distance" | "score";
} = {
  [VectorStoreQueryMode.DEFAULT]: "distance",
  [VectorStoreQueryMode.HYBRID]: "score",
};

// copied verbatim from WeaviateVectorStore
const toWeaviateFilter = (
  collection: Collection,
  standardFilters?: MetadataFilters,
): FilterValue | undefined => {
  if (!standardFilters?.filters.length) return undefined;
  const filtersList = standardFilters.filters.map((filter) =>
    buildFilterItem(collection, filter),
  );
  if (filtersList.length === 1) return filtersList[0]!;
  const condition = standardFilters.condition ?? "and";
  return Filters[condition](...filtersList);
};

// copied verbatim from WeaviateVectorStore
const buildFilterItem = (
  collection: Collection,
  filter: MetadataFilter,
): FilterValue => {
  const { key, operator, value } = filter;

  switch (operator) {
    case "==": {
      return collection.filter.byProperty(key).equal(value);
    }
    case "!=": {
      return collection.filter.byProperty(key).notEqual(value);
    }
    case ">": {
      return collection.filter
        .byProperty(key)
        .greaterThan(parseNumberValue(value));
    }
    case "<": {
      return collection.filter
        .byProperty(key)
        .lessThan(parseNumberValue(value));
    }
    case ">=": {
      return collection.filter
        .byProperty(key)
        .greaterOrEqual(parseNumberValue(value));
    }
    case "<=": {
      return collection.filter
        .byProperty(key)
        .lessOrEqual(parseNumberValue(value));
    }
    case "any": {
      return collection.filter
        .byProperty(key)
        .containsAny(parseArrayValue(value).map(String));
    }
    case "all": {
      return collection.filter
        .byProperty(key)
        .containsAll(parseArrayValue(value).map(String));
    }
    default: {
      throw new Error(`Operator ${operator} is not supported.`);
    }
  }
};


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

  // this is the same as the parent class, EXCEPT that it includes the vector in the results, which I need for the embedding map.
  public async query(
    query: VectorStoreQuery & {
      queryStr: string;
    },
  ): Promise<VectorStoreQueryResult> {
    // @ts-ignore # this method is private but I want to use it anyways!
    const collection = await this.ensureCollection();
    // @ts-ignore # this method is private but I want to use it anyways!
    const allProperties = await this.getAllProperties();

    let filters: FilterValue | undefined = undefined;

    if (query.docIds) {
      filters = collection.filter
        .byProperty("doc_id")
        .containsAny(query.docIds);
    }

    if (query.filters) {
      filters = toWeaviateFilter(collection, query.filters);
    }

    // @ts-ignore # I don't care about BaseHybridOptions
    const hybridOptions: BaseHybridOptions<undefined> = {
      returnMetadata: Object.values(SIMILARITY_KEYS),
      returnProperties: allProperties,
      includeVector: true,
    };
    // @ts-ignore # this method is private but I want to use it anyways!
    const alpha = this.getQueryAlpha(query);
    if (query.queryEmbedding) {
      hybridOptions.vector = query.queryEmbedding;
    }
    if (query.similarityTopK) {
      hybridOptions.limit = query.similarityTopK;
    }
    if (alpha) {
      hybridOptions.alpha = alpha;
    }
    if (filters) {
      hybridOptions.filters = filters;
    }

    const queryResult = await collection.query.hybrid(
      query.queryStr,
      hybridOptions,
    );

    const entries = queryResult.objects;

    const similarityKey = SIMILARITY_KEYS[query.mode];
    const nodes: BaseNode<Metadata>[] = [];
    const similarities: number[] = [];
    const ids: string[] = [];

    entries.forEach((entry: any, index: any) => {
      if (index < query.similarityTopK && entry.metadata) {
        const node = metadataDictToNode(entry.properties);
        
        // this is all that I (Jeremy) added.
        node.embedding = entry.vectors.default;
        // @ts-ignore # this method is private but I want to use it anyways!
        node.setContent(entry.properties[this.contentKey]);
        nodes.push(node);
        ids.push(entry.uuid);
        // @ts-ignore # this method is private but I want to use it anyways!
        similarities.push(this.getNodeSimilarity(entry, similarityKey));
      }
    });

    return {
      nodes,
      similarities,
      ids,
    };
  }  
}



