import { Document } from "llamaindex";
import { readFileSync } from "fs";
import Papa from "papaparse";

/*
 I thought about only loading the user's specified metadataColumns (and omitting any unspecified ones)
 but decided that -- on the principle that users know how to modify CSVs -- it's better to show all 
 columns in the record detail view, assuming that they left the column in the CSV for a reason.

 Metadata column selection is thus about controlling which columns are shown in the results list view only.
*/

// Reserved metadata keys, set only when searching several text columns (see loadDocumentsFromCsvColumns).
export const ROW_KEY = "mf_row"; // index of the CSV row a document came from; used to deduplicate results by row
export const COLUMN_KEY = "mf_column"; // the text column that holds this document's text

function readRecords(filePath: string): any[] {
  const fileContent = readFileSync(filePath, "utf-8");
  const { data: records } = Papa.parse(fileContent, {
    header: true,
    skipEmptyLines: true,
  });
  return records as any[];
}

// For when the user wants to search several (long) text columns: makes one document per row *per
// text column*, so each column is chunked and embedded on its own. Every document keeps all the
// other columns (including the other text columns) as metadata, so that a result can show the
// whole row, and is tagged with its row and column so that results can be deduplicated by row.
// Documents are ordered by row, so that taking a slice of the middle yields whole rows.
export function loadDocumentsFromCsvColumns(
  filePath: string,
  textColumnNames: string[]
): Document[] {
  return readRecords(filePath).flatMap((record: any, row: number) =>
    textColumnNames.map((textColumnName) => {
      const { [textColumnName]: text, ...metadata } = record;
      return new Document({
        text,
        metadata: {
          ...Object.fromEntries(Object.entries(metadata).map(([k, v]) => [k, v ?? ""])),
          [ROW_KEY]: row,
          [COLUMN_KEY]: textColumnName,
        },
      });
    })
  );
}

export async function loadDocumentsFromCsv(
  filePath: string,
  textColumnName: string
  //metadataColumns: string[]
): Promise<Document[]> {
  const records = readRecords(filePath);

  return records.map((record: any) => {
    const { [textColumnName]: text, ...metadata } = record;
    return new Document({
      text,
      metadata: Object.fromEntries(
        Object.entries(metadata).map(([k, v]) => [k, v ?? ""])
        //metadataColumns.map((col) => [col, metadata[col] ?? ""])
      ),
    });
  });
}