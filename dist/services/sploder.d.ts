import { TextNode, TransformComponent } from "llamaindex";
interface SploderConfig {
    maxStringTokenCount: number;
}
export declare class Sploder extends TransformComponent {
    private maxTokenCount;
    private tokenizer;
    constructor(config: SploderConfig);
    private getTokenCount;
    transform(nodes: TextNode[]): Promise<TextNode[]>;
}
export {};
//# sourceMappingURL=sploder.d.ts.map