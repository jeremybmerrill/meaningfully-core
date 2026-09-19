// Vendored copy of @llamaindex/bm25-retriever (github.com/run-llama/LlamaIndexTS,
// packages/providers/retriever/bm25), patched for a CJS/ESM interop bug in its dependency
// "okapibm25". okapibm25's compiled CJS output does `exports.default = BM25` (with a
// non-enumerable `__esModule` marker) but never reassigns `module.exports`. That's fine for
// `require()`, but under Node's native ESM/CJS interop, a plain `import BM25 from "okapibm25"`
// resolves the default import to `module.exports` itself (the whole exports object, not
// `.default`), so calling it throws "BM25 is not a function". We publish this package to npm
// as a dependency of downstream apps, so we can't rely on patch-package (its postinstall step
// doesn't run for those consumers) -- vendoring lets us normalize the interop ourselves instead.
import okapibm25 from "okapibm25";
import {
  BaseRetriever,
  MetadataMode,
  extractText,
  type BaseNode,
  type BaseDocumentStore,
  type NodeWithScore,
  type QueryBundle,
} from "llamaindex";

// See the module-level comment above for why this unwrapping is necessary.
const BM25 = okapibm25.default;

export type Bm25RetrieverOptions = {
  docStore: BaseDocumentStore;
  nodes?: BaseNode[];
  topK?: number;
  docIds?: string[];
};

export class Bm25Retriever extends BaseRetriever {
  private docStore: BaseDocumentStore;
  private nodes?: BaseNode[];
  private topK: number;
  private docIds: string[];

  constructor(options: Bm25RetrieverOptions) {
    super();
    this.docStore = options.docStore;
    this.nodes = options.nodes;
    this.topK = options.topK ?? 10;
    this.docIds = options.docIds ?? [];
  }

  async _retrieve(params: QueryBundle): Promise<NodeWithScore[]> {
    const { query } = params;
    const queryStr = extractText(query);

    let nodes;
    if (this.nodes) {
      nodes = this.nodes;
    } else if (this.docIds.length) {
      nodes = (
        await Promise.all(this.docIds.map((id) => this.docStore.getDocument(id, false)))
      ).filter((node): node is NonNullable<typeof node> => !!node);
    } else {
      nodes = Object.values(await this.docStore.docs());
    }

    const contents = nodes.map((node) => node.getContent(MetadataMode.NONE) || "");
    const scores = BM25(contents, queryStr.toLowerCase().split(/\s+/)) as number[];

    const scoredNodes = nodes.map((node, i) => ({
      node,
      score: scores[i] || 0,
    }));
    scoredNodes.sort((a, b) => b.score - a.score);
    return scoredNodes.slice(0, this.topK);
  }
}
