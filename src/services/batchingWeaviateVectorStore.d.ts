import { WeaviateVectorStore } from '@llamaindex/weaviate';
import { BaseNode } from 'llamaindex';
export declare class BatchingWeaviateVectorStore extends WeaviateVectorStore {
    add(nodes: BaseNode[]): Promise<string[]>;
}
