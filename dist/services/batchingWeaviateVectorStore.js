import { WeaviateVectorStore } from '@llamaindex/weaviate';
/*

Patched version of WeaviateVectorStore to handle large batches by splitting into smaller chunks.

When I loaded a large-ish (5.4MB) spreadsheet, I got a Weaviate error about trying to load too much data at once.

*/
export class BatchingWeaviateVectorStore extends WeaviateVectorStore {
    async add(nodes) {
        const batchSize = 100; // Define the batch size
        const results = []; // Collect results from each batch
        for (let i = 0; i < nodes.length; i += batchSize) {
            const batch = nodes.slice(i, i + batchSize);
            const batchResults = await super.add(batch); // Call the parent class's add method for each batch
            results.push(...batchResults); // Aggregate results
        }
        return results; // Return aggregated results
    }
}
//# sourceMappingURL=batchingWeaviateVectorStore.js.map