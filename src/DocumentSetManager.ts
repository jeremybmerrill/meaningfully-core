import Database from 'better-sqlite3';
import type { Database as SqliteDatabase } from 'better-sqlite3';
import type { DocumentSetMetadata, Settings } from './types/index.js';
import { join } from 'path';

export class DocumentSetManager {
  private sqliteDb: SqliteDatabase;

  constructor(storagePath: string) {
    // Initialize SQLite database for document set metadata
    this.sqliteDb = new Database(join(storagePath, 'metadata.db'));
    
    // Initialize the collection
    this.initializeDatabase();
  }

  private async initializeDatabase() {
    // Create SQLite table for document sets
    this.sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS document_sets (
        set_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        upload_date TEXT NOT NULL,
        parameters TEXT NOT NULL,
        total_documents INTEGER NOT NULL DEFAULT 0
      );
    `);
    this.sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS meaningfully_settings (
        settings_id INTEGER PRIMARY KEY AUTOINCREMENT,
        settings TEXT NOT NULL
      );
    `);    
  }

  async addDocumentSet(metadata: Omit<DocumentSetMetadata, 'documentSetId'>): Promise<number> {
    const stmt = this.sqliteDb.prepare(`
      INSERT INTO document_sets (name, upload_date, parameters, total_documents)
      VALUES (?, ?, ?, ?)
    `);

    const result = stmt.run(
      metadata.name,
      metadata.uploadDate.toISOString(),
      JSON.stringify(metadata.parameters),
      metadata.totalDocuments
    );

    return result.lastInsertRowid as number;
  }

  async getDocumentSet(documentSetId: number): Promise<DocumentSetMetadata | null> {
    const stmt = this.sqliteDb.prepare(`
      SELECT * FROM document_sets WHERE set_id = ?
    `);

    const row = stmt.get(documentSetId) as {
      set_id: number;
      name: string;
      upload_date: string;
      parameters: string;
      total_documents: number;
    } | undefined;
    if (!row) return null;

    return {
      documentSetId: row.set_id,
      name: row.name,
      uploadDate: new Date(row.upload_date),
      parameters: JSON.parse(row.parameters),
      totalDocuments: row.total_documents
    };
  }

  async getDocumentSets(page: number = 1, pageSize: number = 10): Promise<{documents: DocumentSetMetadata[], total: number}> {
    const offset = (page - 1) * pageSize;
    
    // Get total count
    const countStmt = this.sqliteDb.prepare('SELECT COUNT(*) as count FROM document_sets');
    const totalCountRow = countStmt.get() as { count: number };
    const totalCount = totalCountRow.count;

    // Get paginated results
    const stmt = this.sqliteDb.prepare(`
        SELECT * FROM document_sets 
        ORDER BY upload_date DESC 
        LIMIT ? OFFSET ?
    `);
    const rows = stmt.all(pageSize, offset) as Array<{
      set_id: number;
      name: string;
      upload_date: string;
      parameters: string;
      total_documents: number;
    }> ;

    const documents = rows.map((row) => ({
        documentSetId: row.set_id,
        name: row.name,
        uploadDate: new Date(row.upload_date),
        parameters: JSON.parse(row.parameters),
        totalDocuments: row.total_documents
    }));

    return {
        documents,
        total: totalCount
    };
  }

  async updateDocumentCount(documentSetId: number, count: number) {
    const stmt = this.sqliteDb.prepare(`
      UPDATE document_sets 
      SET total_documents = total_documents + ?
      WHERE set_id = ?
    `);
    
    stmt.run(count, documentSetId);
  }

  async deleteDocumentSet(documentSetId: number) {
    const stmt = this.sqliteDb.prepare(`
      DELETE FROM document_sets
      WHERE set_id = ?
    `);
    
    stmt.run(documentSetId);
  }

  async getSettings() { 
    const DEFAULT_SETTINGS = {
      "openAIKey": null,
      "oLlamaBaseURL": null,
      "azureOpenAIKey": null,
      "azureOpenAIEndpoint": null,
      "azureOpenAIApiVersion": "2024-02-01",
      "mistralApiKey": null,
      "geminiApiKey": null,
    }
    const stmt = this.sqliteDb.prepare(`
      SELECT * FROM meaningfully_settings WHERE settings_id = 1
    `);
    
    const row = stmt.get() as { settings: string } | undefined;
    let settings;
    if (row){
      try {
        settings = JSON.parse(row.settings) as Settings;
      } catch (error) {
        console.error("Error parsing settings JSON:", error);
        settings = DEFAULT_SETTINGS;
      }
    }else{
      settings = DEFAULT_SETTINGS;
    }
    settings = Object.assign({}, DEFAULT_SETTINGS, settings)
    return settings; 
  }

  async setSettings(settings: Settings){
    const stmt = this.sqliteDb.prepare(`
      INSERT OR REPLACE INTO meaningfully_settings (settings_id, settings)
      VALUES (1, ?)
    `);
    
    stmt.run(JSON.stringify(settings));
    return Object.assign(settings, {"success": true});
  }

  close() {
    this.sqliteDb.close();
  }
}