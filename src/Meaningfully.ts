import { MetadataManager } from './MetadataManager.js';
import { loadDocumentsFromCsv } from './services/csvLoader.js';
import { createEmbeddings, getIndex, search, previewResults, getDocStore } from './api/embedding.js';
import { sanitizeProjectName, capitalizeFirstLetter } from "./utils.js";
import { join } from 'path';
import type { DocumentSetParams, Settings, MetadataFilter, Clients } from './types/index.js';
import fs from 'fs';

type HasFilePath = {filePath: string};
type DocumentSetParamsFilePath = DocumentSetParams & HasFilePath;

const MASKING_PREFIX_LENGTH = 8; // how many characters to show at the start and end of an API key when masking it for display
                                  // Gemini API keys are 39 chars; Mistral is 32, so MASKING_PREFIX_LENGTH must be < 16 for ANYTHING to be masked.
const maskKey = (key: string | null, n: number = MASKING_PREFIX_LENGTH): string | null => {
  if (!key) return null;
  return (key.length > (n*2)) ? key.slice(0, n) + "*******" + key.slice(key.length - n) : key;
};


export class MeaningfullyAPI {
  private metadataManager: MetadataManager;
  private storagePath: string;
  private clients: Clients;

  constructor({ storagePath, weaviateClient, postgresClient, metadataManager }: { storagePath: string, weaviateClient?: any, postgresClient?: any, metadataManager: MetadataManager }) {
    this.storagePath = storagePath;
    this.metadataManager = metadataManager;
    this.clients = {
      weaviateClient: weaviateClient,
      postgresClient: postgresClient
    };
  }

  setClients(clients: Clients) {
    this.clients = { ...this.clients, ...clients };
  }
  getClients() {
    return this.clients;
  }

  async listDocumentSets(page: number = 1, pageSize: number = 10) {
    return await this.metadataManager.getDocumentSets(page, pageSize);
  }

  async getDocumentSet(documentSetId: number) {
    return await this.metadataManager.getDocumentSet(documentSetId);
  }


  async deleteDocumentSet(documentSetId: number) {
    // Delete the document set from the database
    const result = await this.metadataManager.getDocumentSet(documentSetId);

    if (result){
      // Delete the document set from the database
      await this.metadataManager.deleteDocumentSet(documentSetId);
      // Delete the associated files from the filesystem
      if (result.parameters.vectorStoreType === 'postgres'){
        if (this.clients.postgresClient) {
          try {
            await this.deletePostgresVectorStore(result.name)
            await this.deletePostgresIndexStore(result.name)
            await this.deletePostgresDocStore(result.name)
          } catch (error) {
            console.error(`Error deleting Postgres tables for ${sanitizeProjectName(result.name)}`, error);
          }
        }
      } else if (result.parameters.vectorStoreType === 'weaviate'){
        if (this.clients.weaviateClient) {
          try {
            await this.deleteWeaviateVectorStore(result.name);
          } catch (error) {
            console.error("Error deleting Weaviate class:", error);
          }
        } else {
          this.deleteSimpleVectorStore(result.name);
        }
        // Remove the directory and its contents
        this.deleteSimpleDocStore(result.name);
        this.deleteSimpleIndexStore(result.name);
      } else {
        this.deleteSimpleDocStore(result.name);
        this.deleteSimpleIndexStore(result.name);
        this.deleteSimpleVectorStore(result.name);
      }
    }
    return { success: true };
  }

  // TODO: this is an awful hack.
  // I don't really want to have it be an argument to the constructor
  // maybe I should make clients accept EITHER, not both?
  getVectorStoreType() {
    return this.clients.postgresClient ? 'postgres' : (this.clients.weaviateClient ? 'weaviate' : 'simple');
  }

  async generatePreviewData(data: DocumentSetParamsFilePath) {
    const vectorStoreType = this.getVectorStoreType();


    try {
      if (!data.textColumns[0]) {
        throw new Error("No text column specified for preview.");
      }
      const documents = await loadDocumentsFromCsv(data.filePath, data.textColumns[0] as string);
      if (documents.length === 0) {
        return {
          success: false,
          error: "That CSV does not appear to contain any documents. Please check the file and try again.",
        };
      }      
      return await previewResults(documents, {
        modelName: data.modelName, // needed to tokenize, estimate costs
        modelProvider: data.modelProvider,
        splitIntoSentences: data.splitIntoSentences,
        combineSentencesIntoChunks: data.combineSentencesIntoChunks,
        sploderMaxSize: 100,
        vectorStoreType: vectorStoreType,
        projectName: data.datasetName,
        storagePath: this.storagePath,
        chunkSize: data.chunkSize,
        chunkOverlap: data.chunkOverlap
      });
  } catch (error) {
    throw error;
  }
}

  async uploadCsv(data: DocumentSetParamsFilePath) {
    // figure out if weaviate is available
    const vectorStoreType = this.getVectorStoreType();
    // First create the document set record
    const documentSetId = await this.metadataManager.addDocumentSet({
      name: data.datasetName,
      uploadDate: new Date(),
      parameters: {
        description: data.description,
        textColumns: data.textColumns,
        metadataColumns: data.metadataColumns,
        splitIntoSentences: data.splitIntoSentences,
        combineSentencesIntoChunks: data.combineSentencesIntoChunks,
        sploderMaxSize: data.sploderMaxSize,
        chunkSize: data.chunkSize,
        chunkOverlap: data.chunkOverlap,
        modelName: data.modelName,
        modelProvider: data.modelProvider,
        vectorStoreType: vectorStoreType,
      },
      totalDocuments: 0 // We'll update this after processing
    });

    const embedSettings = await this.metadataManager.getSettings()

    // Load and process the documents
    try {
      // Process each text column
      for (const textColumn of data.textColumns) {
        const documents = await loadDocumentsFromCsv(data.filePath, textColumn);
        
        if (documents.length === 0) {
          console.timeEnd("createEmbeddings Run Time");
          return {
            success: false,
            error: "That CSV does not appear to contain any documents. Please check the file and try again.",
          };
        }

        // Update total documents count
        await this.metadataManager.updateDocumentCount(documentSetId, documents.length);

        // Create embeddings for this column
        let ret = await createEmbeddings(documents, {
          modelName: data.modelName,
          modelProvider: data.modelProvider,
          splitIntoSentences: data.splitIntoSentences,
          combineSentencesIntoChunks: data.combineSentencesIntoChunks,
          sploderMaxSize: 100, // TODO: make configurable
          vectorStoreType: vectorStoreType,
          projectName: data.datasetName,
                        // via https://medium.com/cameron-nokes/how-to-store-user-data-in-electron-3ba6bf66bc1e
          storagePath:  this.storagePath,
          chunkSize: data.chunkSize,
          chunkOverlap: data.chunkOverlap,
        }, embedSettings, this.clients);
        if (!ret.success) {
          throw new Error(ret.error);
        }
      }
      return { success: true, documentSetId };
    } catch (error) {
      // If something fails, we should probably delete the document set
      await this.metadataManager.deleteDocumentSet(documentSetId);
      console.error("deleting document set due to failure ", documentSetId, error);
      throw error;
    }
  }


  async searchDocumentSet(documentSetId: number, query: string, n_results: number = 10,   filters?: MetadataFilter[]  ) {
    const documentSet = await this.metadataManager.getDocumentSet(documentSetId);
    const settings = await this.metadataManager.getSettings();
    if (!documentSet) {
      throw new Error('Document set not found');
    } 
    const index = await getIndex({
      modelName: documentSet.parameters.modelName as string,
      modelProvider: documentSet.parameters.modelProvider as string,
      splitIntoSentences: documentSet.parameters.splitIntoSentences as boolean,
      combineSentencesIntoChunks: documentSet.parameters.combineSentencesIntoChunks as boolean,
      sploderMaxSize: 100,
      vectorStoreType: documentSet.parameters.vectorStoreType as 'simple' | 'weaviate',
      projectName: documentSet.name,
      storagePath: this.storagePath,
      chunkSize: 1024, // not actually used, we just re-use a config object that has this option
      chunkOverlap: 20, // not actually used, we just re-use a config object that has this option
    }, settings, this.clients);
    const results = await search(index, query, n_results, filters);
    return results;
  }   

  async getDocument(documentSetId: number, documentNodeId: string){
    const documentSet = await this.metadataManager.getDocumentSet(documentSetId);
    const settings = await this.metadataManager.getSettings();
    if (!documentSet) {
      throw new Error('Document set not found');
    } 
    const docStore = await getDocStore({
      modelName: documentSet.parameters.modelName as string,
      modelProvider: documentSet.parameters.modelProvider as string,
      splitIntoSentences: documentSet.parameters.splitIntoSentences as boolean,
      combineSentencesIntoChunks: documentSet.parameters.combineSentencesIntoChunks as boolean,
      sploderMaxSize: 100,
      vectorStoreType: documentSet.parameters.vectorStoreType as 'simple' | 'weaviate',
      projectName: documentSet.name,
      storagePath: this.storagePath,
      chunkSize: 1024, // not actually used, we just re-use a config object that has this option
      chunkOverlap: 20, // not actually used, we just re-use a config object that has this option
    }, settings, this.clients);
    const document = await docStore.getNode(documentNodeId);
    if (!document) {
      throw new Error('Document not found');
    }
    return document;
  }


  async getSettings() {
    return this.metadataManager.getSettings();
  }
  async setSettings(settings: Settings) {
    return this.metadataManager.setSettings(settings);
  } 

  async getMaskedSettings() {
    const settings = await this.metadataManager.getSettings();
    return {
      openAIKey: maskKey(settings.openAIKey),
      oLlamaBaseURL: settings.oLlamaBaseURL,
      azureOpenAIKey: maskKey(settings.azureOpenAIKey),
      azureOpenAIEndpoint: settings.azureOpenAIEndpoint,
      azureOpenAIApiVersion: settings.azureOpenAIApiVersion,
      mistralApiKey: maskKey(settings.mistralApiKey),
      geminiApiKey: maskKey(settings.geminiApiKey)
    };
  }
  async setMaskedSettings(newSettings: Settings) { 
    const oldSettings = await this.metadataManager.getSettings();
    const settings = {
      ...newSettings,
      openAIKey: newSettings.openAIKey == maskKey(oldSettings.openAIKey) ? oldSettings.openAIKey : newSettings.openAIKey,
      azureOpenAIKey: newSettings.azureOpenAIKey == maskKey(oldSettings.azureOpenAIKey) ? oldSettings.azureOpenAIKey : newSettings.azureOpenAIKey,
      mistralApiKey: newSettings.mistralApiKey == maskKey(oldSettings.mistralApiKey) ? oldSettings.mistralApiKey : newSettings.mistralApiKey,
      geminiApiKey: newSettings.geminiApiKey == maskKey(oldSettings.geminiApiKey) ? oldSettings.geminiApiKey : newSettings.geminiApiKey
    };
    return this.metadataManager.setSettings(settings);
  }


  // these should be moved to another file, just because they're too low-level for this one.
  async deletePostgresVectorStore(projectName: string) {
    if (this.clients.postgresClient) {
      try {
        await this.clients.postgresClient.query('DROP TABLE IF EXISTS vecs_' + sanitizeProjectName(projectName));
      } catch (error) {
        console.error(`Error deleting Postgres tables for ${sanitizeProjectName(projectName)}`, error);
      }
    }
  }
  async deletePostgresIndexStore(projectName: string) {
    if (this.clients.postgresClient) {
      try {
        await this.clients.postgresClient.query('DROP TABLE IF EXISTS idx_' + sanitizeProjectName(projectName ));
      } catch (error) {
        console.error(`Error deleting Postgres tables for ${sanitizeProjectName(projectName)}`, error);
      }
    }
  }
  async deletePostgresDocStore(projectName: string) {
    if (this.clients.postgresClient) {
      try {
        await this.clients.postgresClient.query('DROP TABLE IF EXISTS docs_' + sanitizeProjectName(projectName));
      } catch (error) {
        console.error(`Error deleting Postgres tables for ${sanitizeProjectName(projectName)}`, error);
      }
    }

  }
  async deleteWeaviateVectorStore(projectName: string) {
    if (this.clients.weaviateClient) {
      await this.clients.weaviateClient.collections.delete(capitalizeFirstLetter(projectName));
    }
        // fs.rmSync(join(this.storagePath, 'weaviate_data', capitalizeFirstLetter(result.name)), { recursive: true, force: true });
  }
  async deleteSimpleVectorStore(projectName: string) {
    fs.rmSync(join(this.storagePath, projectName), { recursive: true, force: true });

  }
  async deleteSimpleDocStore(projectName: string) {
    fs.rmSync(join(this.storagePath, projectName), { recursive: true, force: true });

  }
  async deleteSimpleIndexStore(projectName: string) {
    fs.rmSync(join(this.storagePath, projectName), { recursive: true, force: true });
  }
}
