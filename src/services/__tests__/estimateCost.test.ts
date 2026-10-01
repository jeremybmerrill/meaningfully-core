import { describe, it, expect } from 'vitest';
import { TextNode } from 'llamaindex';
import { estimateCost } from '../embeddings.js';

describe('estimateCost', () => {
  const model = 'text-embedding-3-small';

  it('counts only the node text when all metadata is excluded from embedding', () => {
    const plain = new TextNode({ text: 'Document 1' });
    const withExcludedMetadata = new TextNode({
      text: 'Document 1',
      metadata: { title: 'A much longer title than the text' },
      excludedEmbedMetadataKeys: ['title'],
    });

    expect(estimateCost([withExcludedMetadata], model).tokenCount)
      .toBe(estimateCost([plain], model).tokenCount);
  });

  it('counts embedded metadata columns (metadata not excluded from embedding)', () => {
    const plain = new TextNode({ text: 'Document 1' });
    const withExtraColumn = new TextNode({
      text: 'Document 1',
      metadata: { title: 'A much longer title than the text' },
      excludedEmbedMetadataKeys: [],
    });

    expect(estimateCost([withExtraColumn], model).tokenCount)
      .toBeGreaterThan(estimateCost([plain], model).tokenCount);
  });

  it('computes price from the token count and the model price', () => {
    const { tokenCount, pricePer1M, estimatedPrice } = estimateCost([new TextNode({ text: 'Document 1' })], model);
    expect(pricePer1M).toBe(0.02);
    expect(estimatedPrice).toBeCloseTo(tokenCount * 0.02 / 1_000_000);
  });
});
