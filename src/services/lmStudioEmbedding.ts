import { BaseEmbedding } from "llamaindex";

// LM Studio's OpenAI-compatible /v1/embeddings endpoint doesn't honor "encoding_format": it
// always returns plain float arrays. The `openai` SDK (used by @llamaindex/openai's
// OpenAIEmbedding) defaults to requesting "encoding_format": "base64" and then base64-decodes
// whatever comes back, which silently corrupts LM Studio's plain-float response into all-zero
// vectors -- every input ends up with the same (empty) embedding. Requesting "float" explicitly
// avoids the SDK's base64 path entirely.
export class LMStudioEmbedding extends BaseEmbedding {
  model: string;
  baseURL: string;

  constructor({ model, baseURL }: { model: string; baseURL: string }) {
    super();
    this.model = model;
    this.baseURL = baseURL.replace(/\/$/, "");
  }

  async getTextEmbedding(text: string): Promise<number[]> {
    const response = await fetch(`${this.baseURL}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.model, input: text, encoding_format: "float" }),
    });
    if (!response.ok) {
      throw new Error(`LM Studio embeddings request failed: ${response.status} ${response.statusText}`);
    }
    const { data } = (await response.json()) as { data: { embedding: number[] }[] };
    return data[0]!.embedding;
  }
}
