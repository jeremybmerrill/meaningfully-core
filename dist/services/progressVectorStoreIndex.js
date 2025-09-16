import { VectorStoreIndex, storageContextFromDefaults, IndexDict } from "llamaindex";
import { splitNodesByType } from "llamaindex";
import { addNodesToVectorStores } from "llamaindex";
// Subclass VectorStoreIndex to handle progressCallback
// @ts-ignore
export class ProgressVectorStoreIndex extends VectorStoreIndex {
    static async init(options) {
        const storageContext = options.storageContext ?? (await storageContextFromDefaults({}));
        const indexStore = storageContext.indexStore;
        const docStore = storageContext.docStore;
        // @ts-ignore 
        let indexStruct = await VectorStoreIndex.setupIndexStructFromStorage(indexStore, options);
        if (!options.nodes && !indexStruct) {
            throw new Error("Cannot initialize VectorStoreIndex without nodes or indexStruct");
        }
        indexStruct = indexStruct ?? new IndexDict();
        // @ts-ignore
        const index = new this({
            storageContext,
            docStore,
            indexStruct,
            indexStore,
            vectorStores: options.vectorStores,
        });
        if (options.nodes) {
            // If nodes are passed in, then we need to update the index
            await index.buildIndexFromNodes(options.nodes, {
                logProgress: options.logProgress,
                progressCallback: options.progressCallback,
            });
        }
        return index;
    }
    async buildIndexFromNodes(nodes, options) {
        await this.insertNodes(nodes, options);
    }
    async insertNodes(nodes, options) {
        if (!nodes || nodes.length === 0) {
            return;
        }
        nodes = await this.getNodeEmbeddingResults(nodes, options);
        await addNodesToVectorStores(nodes, this.vectorStores, this.insertNodesToStore.bind(this));
        await this.indexStore.addIndexStruct(this.indexStruct);
    }
    async getNodeEmbeddingResults(nodes, options) {
        const nodeMap = splitNodesByType(nodes);
        for (const type in nodeMap) {
            const nodes = nodeMap[type];
            const embedModel = this.vectorStores[type]?.embedModel ?? this.embedModel;
            if (embedModel && nodes) {
                await embedModel(nodes, {
                    logProgress: options?.logProgress,
                    progressCallback: options?.progressCallback, // Pass progressCallback to embedModel
                });
            }
        }
        return nodes;
    }
}
//# sourceMappingURL=progressVectorStoreIndex.js.map