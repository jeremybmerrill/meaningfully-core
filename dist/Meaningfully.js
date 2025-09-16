import { loadDocumentsFromCsv } from './services/csvLoader.js';
import { createEmbeddings, getIndex, search, previewResults, getDocStore } from './api/embedding.js';
import { capitalizeFirstLetter } from './utils.js';
import { join } from 'path';
import fs from 'fs';
const MASKING_PREFIX_LENGTH = 8; // how many characters to show at the start and end of an API key when masking it for display
// Gemini API keys are 39 chars; Mistral is 32, so MASKING_PREFIX_LENGTH must be < 16 for ANYTHING to be masked.
const maskKey = (key, n = MASKING_PREFIX_LENGTH) => {
    if (!key)
        return null;
    return (key.length > (n * 2)) ? key.slice(0, n) + "*******" + key.slice(key.length - n) : key;
};
export class MeaningfullyAPI {
    manager;
    storagePath;
    clients;
    constructor({ storagePath, weaviateClient, metadataManager }) {
        this.storagePath = storagePath;
        this.manager = metadataManager;
        this.clients = {
            weaviateClient: weaviateClient,
            postgresClient: null
        };
    }
    setClients(clients) {
        this.clients = { ...this.clients, ...clients };
    }
    getClients() {
        return this.clients;
    }
    async listDocumentSets(page = 1, pageSize = 10) {
        return await this.manager.getDocumentSets(page, pageSize);
    }
    async getDocumentSet(documentSetId) {
        return await this.manager.getDocumentSet(documentSetId);
    }
    async deleteDocumentSet(documentSetId) {
        // Delete the document set from the database
        const result = await this.manager.getDocumentSet(documentSetId);
        if (result) {
            // Delete the document set from the database
            await this.manager.deleteDocumentSet(documentSetId);
            // Delete the associated files from the filesystem
            fs.rmSync(join(this.storagePath, result.name), { recursive: true, force: true });
            fs.rmSync(join(this.storagePath, 'weaviate_data', capitalizeFirstLetter(result.name)), { recursive: true, force: true });
        }
        return { success: true };
    }
    getVectorStoreType() {
        return this.clients.weaviateClient ? 'weaviate' : 'simple';
    }
    async generatePreviewData(data) {
        const vectorStoreType = this.getVectorStoreType();
        try {
            if (!data.textColumns[0]) {
                throw new Error("No text column specified for preview.");
            }
            return await previewResults(data.filePath, data.textColumns[0], {
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
        }
        catch (error) {
            throw error;
        }
    }
    async uploadCsv(data) {
        // figure out if weaviate is available
        const vectorStoreType = this.getVectorStoreType();
        // First create the document set record
        const documentSetId = await this.manager.addDocumentSet({
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
        const embedSettings = await this.manager.getSettings();
        // Load and process the documents
        try {
            // Process each text column
            for (const textColumn of data.textColumns) {
                const documents = await loadDocumentsFromCsv(data.filePath, textColumn);
                // Update total documents count
                await this.manager.updateDocumentCount(documentSetId, documents.length);
                // Create embeddings for this column
                let ret = await createEmbeddings(data.filePath, textColumn, {
                    modelName: data.modelName,
                    modelProvider: data.modelProvider,
                    splitIntoSentences: data.splitIntoSentences,
                    combineSentencesIntoChunks: data.combineSentencesIntoChunks,
                    sploderMaxSize: 100, // TODO: make configurable
                    vectorStoreType: vectorStoreType,
                    projectName: data.datasetName,
                    // via https://medium.com/cameron-nokes/how-to-store-user-data-in-electron-3ba6bf66bc1e
                    storagePath: this.storagePath,
                    chunkSize: data.chunkSize,
                    chunkOverlap: data.chunkOverlap,
                }, embedSettings, this.clients);
                if (!ret.success) {
                    throw new Error(ret.error);
                }
            }
            return { success: true, documentSetId };
        }
        catch (error) {
            // If something fails, we should probably delete the document set
            await this.manager.deleteDocumentSet(documentSetId);
            console.error("deleting document set due to failure ", documentSetId, error);
            throw error;
        }
    }
    async searchDocumentSet(documentSetId, query, n_results = 10, filters) {
        const documentSet = await this.manager.getDocumentSet(documentSetId);
        const settings = await this.manager.getSettings();
        if (!documentSet) {
            throw new Error('Document set not found');
        }
        const index = await getIndex({
            modelName: documentSet.parameters.modelName,
            modelProvider: documentSet.parameters.modelProvider,
            splitIntoSentences: documentSet.parameters.splitIntoSentences,
            combineSentencesIntoChunks: documentSet.parameters.combineSentencesIntoChunks,
            sploderMaxSize: 100,
            vectorStoreType: documentSet.parameters.vectorStoreType,
            projectName: documentSet.name,
            storagePath: this.storagePath,
            chunkSize: 1024, // not actually used, we just re-use a config object that has this option
            chunkOverlap: 20, // not actually used, we just re-use a config object that has this option
        }, settings, this.clients);
        const results = await search(index, query, n_results, filters);
        return results;
    }
    async getDocument(documentSetId, documentNodeId) {
        const documentSet = await this.manager.getDocumentSet(documentSetId);
        if (!documentSet) {
            throw new Error('Document set not found');
        }
        const docStore = await getDocStore({
            modelName: documentSet.parameters.modelName,
            modelProvider: documentSet.parameters.modelProvider,
            splitIntoSentences: documentSet.parameters.splitIntoSentences,
            combineSentencesIntoChunks: documentSet.parameters.combineSentencesIntoChunks,
            sploderMaxSize: 100,
            vectorStoreType: documentSet.parameters.vectorStoreType,
            projectName: documentSet.name,
            storagePath: this.storagePath,
            chunkSize: 1024, // not actually used, we just re-use a config object that has this option
            chunkOverlap: 20, // not actually used, we just re-use a config object that has this option
        });
        const document = await docStore.getNode(documentNodeId);
        if (!document) {
            throw new Error('Document not found');
        }
        return document;
    }
    async getSettings() {
        return this.manager.getSettings();
    }
    async setSettings(settings) {
        return this.manager.setSettings(settings);
    }
    async getMaskedSettings() {
        const settings = await this.manager.getSettings();
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
    async setMaskedSettings(newSettings) {
        const oldSettings = await this.manager.getSettings();
        const settings = {
            ...newSettings,
            openAIKey: newSettings.openAIKey == maskKey(oldSettings.openAIKey) ? oldSettings.openAIKey : newSettings.openAIKey,
            azureOpenAIKey: newSettings.azureOpenAIKey == maskKey(oldSettings.azureOpenAIKey) ? oldSettings.azureOpenAIKey : newSettings.azureOpenAIKey,
            mistralApiKey: newSettings.mistralApiKey == maskKey(oldSettings.mistralApiKey) ? oldSettings.mistralApiKey : newSettings.mistralApiKey,
            geminiApiKey: newSettings.geminiApiKey == maskKey(oldSettings.geminiApiKey) ? oldSettings.geminiApiKey : newSettings.geminiApiKey
        };
        return this.manager.setSettings(settings);
    }
}
//# sourceMappingURL=Meaningfully.js.map