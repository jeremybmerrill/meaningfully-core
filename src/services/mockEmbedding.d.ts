import { BaseEmbedding } from "llamaindex";
export declare class MockEmbedding extends BaseEmbedding {
    constructor();
    getTextEmbedding(text: string): Promise<number[]>;
}
