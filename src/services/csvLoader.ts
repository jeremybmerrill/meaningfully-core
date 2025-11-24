import { Document } from "llamaindex";
import { readFileSync } from "fs";
import Papa from "papaparse";

/*
 I thought about only loading the user's specified metadataColumns (and omitting any unspecified ones)
 but decided that -- on the principle that users know how to modify CSVs -- it's better to show all 
 columns in the record detail view, assuming that they left the column in the CSV for a reason.

 Metadata column selection is thus about controlling which columns are shown in the results list view only.
*/

export async function loadDocumentsFromCsv(
  filePath: string,
  textColumnName: string
  //metadataColumns: string[]
): Promise<Document[]> {
  const fileContent = readFileSync(filePath, "utf-8");
  const { data: records } = Papa.parse(fileContent, {
    header: true,
    skipEmptyLines: true,
  });

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