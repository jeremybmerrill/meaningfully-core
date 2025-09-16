//@ts-nocheck
import { BaseEmbedding } from "llamaindex";
export class MockEmbedding extends BaseEmbedding {
    constructor() {
        super();
    }
    async getTextEmbedding(text) {
        return new Promise((resolve) => {
            resolve([1, 0, 0, 0, 0, 0]);
        });
    }
}
;
//# sourceMappingURL=mockEmbedding.js.map