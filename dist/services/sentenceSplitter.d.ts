import { SentenceSplitter } from "llamaindex";
import natural from "natural";
type TextSplitterFn = (text: string) => string[];
export declare class CustomSentenceSplitter extends SentenceSplitter {
    #private;
    chunkingTokenizerFn: () => TextSplitterFn;
    abbreviations: string[];
    tokenizer: natural.SentenceTokenizer;
    constructor(params?: {
        chunkSize?: number;
        chunkOverlap?: number;
        abbreviations?: string[];
    });
    _splitText(text: string, chunkSize: number): string[];
}
export {};
//# sourceMappingURL=sentenceSplitter.d.ts.map