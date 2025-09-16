import { TextNode, TransformComponent } from "llamaindex";
import { encodingForModel } from "js-tiktoken";
export class Sploder extends TransformComponent {
    maxTokenCount;
    tokenizer; // js-tiktoken encoder
    // TODO: this is a hack to get the tokenizer for the embedding model
    // TODO: this should be a singleton
    constructor(config) {
        super(async (nodes) => nodes); // no-op, to be replaced later
        this.maxTokenCount = config.maxStringTokenCount;
        this.tokenizer = encodingForModel("text-embedding-3-small");
    }
    getTokenCount(text) {
        return this.tokenizer.encode(text).length;
    }
    async transform(nodes) {
        const newNodes = [];
        nodes.forEach((node, index) => {
            // Keep original node
            newNodes.push(node);
            // Skip if text is too long
            if (this.getTokenCount(node.text) > this.maxTokenCount) {
                return;
            }
            const prevNode = index > 0 ? nodes[index - 1] : null;
            const nextNode = index < nodes.length - 1 ? nodes[index + 1] : null;
            // Create node with current + next if available
            if (nextNode) {
                newNodes.push(new TextNode({
                    text: node.text + " " + nextNode.text,
                    metadata: { ...node.metadata, isExpanded: true }
                }));
            }
            // Create node with prev + current + next if both available
            if (prevNode && nextNode) {
                newNodes.push(new TextNode({
                    text: prevNode.text + " " + node.text + " " + nextNode.text,
                    metadata: { ...node.metadata, isExpanded: true }
                }));
            }
        });
        return newNodes;
    }
}
//# sourceMappingURL=sploder.js.map