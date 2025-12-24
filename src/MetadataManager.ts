import type { DocumentSetMetadata, Settings } from './types/index.js';
import type { Knex } from 'knex';

export abstract class MetadataManager {
  protected abstract knex: Knex;

  protected abstract initializeDatabase(): Promise<void>;
  protected abstract close(): void;

  async addDocumentSet(metadata: Omit<DocumentSetMetadata, 'documentSetId'>): Promise<number> {
    const [result] = await this.knex('document_sets')
      .insert({
        name: metadata.name,
        upload_date: metadata.uploadDate.toISOString(), // coerce to ISO string for Sqlite3, which otherwise prefers to store dates as timestamps.
        parameters: JSON.stringify(metadata.parameters),
        total_documents: metadata.totalDocuments
      })
      .returning('set_id');
    
    return typeof result === 'object' ? result.set_id : result;
  }

  async getDocumentSet(documentSetId: number): Promise<DocumentSetMetadata | null> {
    const row = await this.knex('document_sets')
      .where('set_id', documentSetId)
      .first();

    if (!row) return null;

    return {
      documentSetId: row.set_id,
      name: row.name,
      uploadDate: new Date(row.upload_date),
      parameters: JSON.parse(row.parameters),
      totalDocuments: row.total_documents
    };
  }

  async getDocumentSets(page: number = 1, pageSize: number = 10): Promise<{ documents: DocumentSetMetadata[]; total: number }> {
    const offset = (page - 1) * pageSize;
    
    const totalCountRow = await this.knex('document_sets').count('* as count').first();
    const totalCount = totalCountRow ? Number(totalCountRow.count) : 0;

    const rows = await this.knex('document_sets')
      .select('*')
      .orderBy('upload_date', 'desc')
      .limit(pageSize)
      .offset(offset);

    const documents = rows.map((row: any) => ({
      documentSetId: row.set_id,
      name: row.name,
      uploadDate: new Date(row.upload_date),
      parameters: JSON.parse(row.parameters),
      totalDocuments: row.total_documents
    }));

    return { documents, total: totalCount };
  }

  async updateDocumentCount(documentSetId: number, count: number): Promise<void> {
    await this.knex('document_sets')
      .where('set_id', documentSetId)
      .increment('total_documents', count);
  }

  async deleteDocumentSet(documentSetId: number): Promise<void> {
    await this.knex('document_sets')
      .where('set_id', documentSetId)
      .delete();
  }

  async getSettings(): Promise<Settings> {
    const DEFAULT_SETTINGS: Settings = {
      openAIKey: null,
      oLlamaBaseURL: null,
      azureOpenAIKey: null,
      azureOpenAIEndpoint: null,
      azureOpenAIApiVersion: "2024-02-01",
      mistralApiKey: null,
      geminiApiKey: null,
    };

    const row = await this.knex('meaningfully_settings')
      .where('settings_id', 1)
      .first();
    
    return row ? { ...DEFAULT_SETTINGS, ...JSON.parse(row.settings) } : DEFAULT_SETTINGS;
  }

  async setSettings(settings: Settings): Promise<{ success: boolean }> {
    await this.knex('meaningfully_settings')
      .insert({
        settings_id: 1,
        settings: JSON.stringify(settings)
      })
      .onConflict('settings_id')
      .merge();
    
    return { success: true };
  }
}
