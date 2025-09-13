import { VectorStoreIndex, VectorIndexOptions as BaseVectorIndexOptions } from "llamaindex";
import { BaseNode } from "llamaindex";
export interface VectorIndexOptions extends BaseVectorIndexOptions {
    progressCallback?: (progress: number, total: number) => void;
}
export declare class ProgressVectorStoreIndex extends VectorStoreIndex {
    static init(options: VectorIndexOptions): Promise<VectorStoreIndex>;
    buildIndexFromNodes(nodes: BaseNode[], options?: {
        logProgress?: boolean;
        progressCallback?: (progress: number, total: number) => void;
    }): Promise<void>;
    insertNodes(nodes: BaseNode[], options?: {
        logProgress?: boolean;
        progressCallback?: (progress: number, total: number) => void;
    }): Promise<void>;
    getNodeEmbeddingResults(nodes: BaseNode[], options?: {
        logProgress?: boolean;
        progressCallback?: (progress: number, total: number) => void;
    }): Promise<BaseNode[]>;
}
