import { transformDocumentsToNodes, estimateCost, searchDocuments, searchDocumentsBm25, getExistingVectorStoreIndex, persistNodes, persistDocuments, getStorageContext } from "../services/embeddings.js";
import type { EmbeddingConfig, EmbeddingResult, SearchResponse, PreviewResult, SampleDocument, Settings, MetadataFilter, Clients } from "../types/index.js";
import { loadDocumentsFromCsv } from "../services/csvLoader.js";
import { MetadataMode, Document, type BaseDocumentStore, type NodeWithScore } from "llamaindex";
import { ProgressManager } from "../services/progressManager.js";

function toSearchResponse(results: NodeWithScore[], hasMore: boolean): SearchResponse {
  return {
    results: results.map((result: any) => ({
      text: result.node.getContent(MetadataMode.NONE),
      score: result.score ?? 0,
      metadata: result.node.metadata,
      //  @ts-ignore
      sourceNodeId: result.node.relationships?.SOURCE?.nodeId
    })),
    hasMore
  };
}

// Take 10 rows from the middle of the dataset for preview.
// We take a consistent 10 so that the results of the preview are consistent (i.e. with a
// larger chunk size, you have fewer, longer results, but more shorter ones if you adjust it)
// and we take from the middle because the initial rows may be idiosyncratic.
function extractPreviewSample(documents: Document[]): SampleDocument[] {
  return documents
    .slice(Math.floor(documents.length / 2), Math.floor(documents.length / 2) + 10)
    .map((document) => ({ text: document.text, metadata: document.metadata }));
}

export async function createEmbeddings(
  documents: Document[],
  config: EmbeddingConfig,
  settings: Settings,
  clients: Clients
): Promise<EmbeddingResult> {
  try {
    console.time("createEmbeddings Run Time");
    const operationId = `embed-${Date.now()}`;
    const progressManager = ProgressManager.getInstance();
    progressManager.startOperation(operationId, 100);

    progressManager.updateProgress(operationId, 5);
    
    const nodes = await transformDocumentsToNodes(documents, config);
      
    const progressCallback = (progress: number, total: number) => {
        const percentage = Math.floor((progress / total) * 90) + 5; // Map to 5-95% of total progress
        progressManager.updateProgress(operationId, percentage);
      };
    const [index] = await Promise.all([
      persistNodes(nodes, config, settings, clients, progressCallback),
      persistDocuments(documents, config, settings, clients)
    ]);
    
    progressManager.completeOperation(operationId);
    console.timeEnd("createEmbeddings Run Time");
    return {
      success: true,
      index,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    };
  }
}

// TODO: rename this to be parallel to createEmbeddings
// Whole-file: reads every document once, computes an exact price estimate across the full
// dataset, and returns a small representative sample (+ total document count) alongside the
// initial preview, so that later config-only changes (chunk size, model, etc.) can call
// previewSample() instead of re-reading the file from scratch.
export async function previewResults(
  documents: Document[],
  config: EmbeddingConfig
): Promise<PreviewResult> {
  try {
    const previewDocumentsSubset = documents.slice(
      Math.floor(documents.length / 2),
      Math.floor(documents.length / 2) + 10
    );

    const previewNodes = await transformDocumentsToNodes(documents, config);
    const previewSubsetNodes = await transformDocumentsToNodes(previewDocumentsSubset, config);
    const { estimatedPrice, tokenCount, pricePer1M } = estimateCost(previewNodes, config.modelName);

    return {
      success: true,
      nodes: previewSubsetNodes.map((node: any) => ({
        text: node.text,
        metadata: node.metadata
      })),
      estimatedPrice,
      tokenCount,
      pricePer1M,
      documentCount: documents.length,
      sample: extractPreviewSample(documents),
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred"
    };
  }
}

// Sample-only: re-chunks a previously-extracted sample (from previewResults) under a new
// config, without touching the source file. estimatedPrice/tokenCount are extrapolated from
// the sample by the ratio of documentCount to sample size -- this is the trade-off that lets
// reactive config changes (chunk size, splitting, model) skip re-transforming the whole
// dataset; the number is necessarily an approximation, not the exact figure previewResults
// gives you.
export async function previewSample(
  sample: SampleDocument[],
  documentCount: number,
  config: EmbeddingConfig
): Promise<PreviewResult> {
  try {
    if (sample.length === 0) {
      return {
        success: false,
        error: "No sample data available for preview.",
      };
    }

    const sampleDocuments = sample.map((s) => new Document({ text: s.text, metadata: s.metadata }));
    const sampleNodes = await transformDocumentsToNodes(sampleDocuments, config);
    const { estimatedPrice, tokenCount, pricePer1M } = estimateCost(sampleNodes, config.modelName);
    const scale = documentCount / sample.length;

    return {
      success: true,
      nodes: sampleNodes.map((node: any) => ({
        text: node.text,
        metadata: node.metadata
      })),
      estimatedPrice: estimatedPrice * scale,
      tokenCount: Math.round(tokenCount * scale),
      pricePer1M
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred"
    };
  }
}

export async function getDocStore(config: EmbeddingConfig, settings: Settings, clients: Clients) {
  return (await getStorageContext(config, settings, clients)).docStore;
}

export async function getIndex(config: EmbeddingConfig, settings: Settings, clients: Clients) {
  return await getExistingVectorStoreIndex(config, settings, clients);
}

export async function search(
  index: any,
  query: string,
  numResults: number = 10,
  filters?: MetadataFilter[],
  offset: number = 0
): Promise<SearchResponse> {
  const { results, hasMore } = await searchDocuments(index, query, numResults, filters, offset);
  return toSearchResponse(results, hasMore);
}

// BM25 keyword search: ranks the same indexed chunks by keyword relevance rather than
// embedding similarity. Metadata filters aren't supported in this mode -- see
// searchDocumentsBm25.
export async function searchBm25(
  docStore: BaseDocumentStore,
  query: string,
  numResults: number = 10,
  offset: number = 0
): Promise<SearchResponse> {
  const { results, hasMore } = await searchDocumentsBm25(docStore, query, numResults, offset);
  return toSearchResponse(results, hasMore);
}
