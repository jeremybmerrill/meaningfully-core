import type { DocumentSetMetadata, Settings } from './types/index.js';

export abstract class MetadataManager {
  protected queries = {
    /* 
    Note: RETURNING on non-select/non-create statements is important for compatibility between SQLite and PostgreSQL.
    (Without it, better-sqlite would demand to use run() instead of all() or get(), which would break the abstraction.)
    */
    createDocumentSetsTable: `
      CREATE TABLE IF NOT EXISTS document_sets (
        set_id SERIAL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        upload_date TIMESTAMP NOT NULL,
        parameters TEXT NOT NULL,
        total_documents INTEGER NOT NULL DEFAULT 0
      );
    `,
    createSettingsTable: `
      CREATE TABLE IF NOT EXISTS meaningfully_settings (
        settings_id SERIAL PRIMARY KEY,
        settings TEXT NOT NULL
      );
    `,
    insertDocumentSet: `
      INSERT INTO document_sets (name, upload_date, parameters, total_documents)
      VALUES ($1, $2, $3, $4) RETURNING set_id
    `,
    selectDocumentSet: `
      SELECT * FROM document_sets WHERE set_id = $1
    `,
    selectDocumentSets: `
      SELECT * FROM document_sets ORDER BY upload_date DESC LIMIT $1 OFFSET $2
    `,
    countDocumentSets: `
      SELECT COUNT(*) as count FROM document_sets
    `,
    updateDocumentCount: `
      UPDATE document_sets SET total_documents = total_documents + $1 WHERE set_id = $2 RETURNING *
    `,
    deleteDocumentSet: `
      DELETE FROM document_sets WHERE set_id = $1 RETURNING *
    `,
    selectSettings: `
      SELECT * FROM meaningfully_settings WHERE settings_id = 1
    `,
    upsertSettings: `
      INSERT INTO meaningfully_settings (settings_id, settings)
      VALUES (1, $1)
      ON CONFLICT (settings_id) DO UPDATE SET settings = $2
      RETURNING *
    `     
    // the two arguments $1 and $2 are identical, but, to work around a cross-compatibility bug in SQLite versus Postgresql,
    // where PG can accept the same argument twice (specified as $1 in two places), but SQLITE cannot (it just has ? placeholders)
    // they are specified separately.
  };

  protected abstract runQuery<T>(query: string, params?: any[]): Promise<T[]>;
  protected abstract runQuerySingle<T>(query: string, params?: any[]): Promise<T | null>;
  protected abstract initializeDatabase(): Promise<void>;
  protected abstract close(): void;

  async addDocumentSet(metadata: Omit<DocumentSetMetadata, 'documentSetId'>): Promise<number> {
    const result = await this.runQuerySingle<{ set_id: number }>(this.queries.insertDocumentSet, [
      metadata.name,
      metadata.uploadDate.toISOString(),
      JSON.stringify(metadata.parameters),
      metadata.totalDocuments
    ]);
    return result?.set_id || 0;
  }

  async getDocumentSet(documentSetId: number): Promise<DocumentSetMetadata | null> {
    const row = await this.runQuerySingle<{
      set_id: number;
      name: string;
      upload_date: string;
      parameters: string;
      total_documents: number;
    }>(this.queries.selectDocumentSet, [documentSetId]);

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
    const totalCountRow = await this.runQuerySingle<{ count: number }>(this.queries.countDocumentSets);
    const totalCount = totalCountRow?.count || 0;

    const rows = await this.runQuery<{
      set_id: number;
      name: string;
      upload_date: string;
      parameters: string;
      total_documents: number;
    }>(this.queries.selectDocumentSets, [pageSize, offset]);

    const documents = rows.map((row) => ({
      documentSetId: row.set_id,
      name: row.name,
      uploadDate: new Date(row.upload_date),
      parameters: JSON.parse(row.parameters),
      totalDocuments: row.total_documents
    }));

    return { documents, total: totalCount };
  }

  async updateDocumentCount(documentSetId: number, count: number): Promise<void> {
    await this.runQuery(this.queries.updateDocumentCount, [count, documentSetId]);
  }

  async deleteDocumentSet(documentSetId: number): Promise<void> {
    await this.runQuery(this.queries.deleteDocumentSet, [documentSetId]);
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

    const row = await this.runQuerySingle<{ settings: string }>(this.queries.selectSettings);
    return row ? { ...DEFAULT_SETTINGS, ...JSON.parse(row.settings) } : DEFAULT_SETTINGS;
  }

  async setSettings(settings: Settings): Promise<{ success: boolean }> {
    // the JSON.stringify(settings) is repeated to work around a cross-compatibility bug in SQLite versus Postgresql
    // where PG can accept the same argument twice (specified as $1 in two places), but SQLITE cannot (it just has ? placeholders)
    await this.runQuery(this.queries.upsertSettings, [JSON.stringify(settings), JSON.stringify(settings)]);
    return { success: true };
  }
}
