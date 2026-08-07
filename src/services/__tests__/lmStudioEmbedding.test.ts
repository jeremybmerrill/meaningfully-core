import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LMStudioEmbedding } from '../lmStudioEmbedding.js';

describe('LMStudioEmbedding', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  it('requests "encoding_format": "float" so the openai SDK\'s base64 auto-decoding never kicks in', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [{ embedding: [0.1, 0.2, 0.3] }] }),
    });

    const embedModel = new LMStudioEmbedding({ model: 'text-embedding-nomic-embed-text-v1.5', baseURL: 'http://localhost:1234/' });
    await embedModel.getTextEmbedding('hello');

    const [url, opts] = (fetch as any).mock.calls[0];
    expect(url).toBe('http://localhost:1234/v1/embeddings');
    expect(JSON.parse(opts.body)).toMatchObject({ encoding_format: 'float' });
  });

  it('returns different embeddings for different inputs (regression: LM Studio ignores encoding_format and always returns plain floats)', async () => {
    (fetch as any).mockImplementation((url: string, opts: any) => {
      const { input } = JSON.parse(opts.body);
      const embedding = input === 'cat' ? [0.1, 0.2, 0.3] : [0.9, 0.8, 0.7];
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ data: [{ embedding }] }) });
    });

    const embedModel = new LMStudioEmbedding({ model: 'text-embedding-nomic-embed-text-v1.5', baseURL: 'http://localhost:1234' });
    const catEmbedding = await embedModel.getTextEmbedding('cat');
    const dogEmbedding = await embedModel.getTextEmbedding('dog');

    expect(catEmbedding).toEqual([0.1, 0.2, 0.3]);
    expect(dogEmbedding).toEqual([0.9, 0.8, 0.7]);
    expect(catEmbedding).not.toEqual(dogEmbedding);
  });

  it('throws when the LM Studio server is unreachable or returns an error', async () => {
    (fetch as any).mockResolvedValue({ ok: false, status: 500, statusText: 'Internal Server Error' });

    const embedModel = new LMStudioEmbedding({ model: 'text-embedding-nomic-embed-text-v1.5', baseURL: 'http://localhost:1234' });
    await expect(embedModel.getTextEmbedding('hello')).rejects.toThrow('LM Studio embeddings request failed');
  });
});
