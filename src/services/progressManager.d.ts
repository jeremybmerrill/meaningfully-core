/**
 * A simple manager to track progress of various operations
 */
export declare class ProgressManager {
    private static instance;
    private progressMap;
    private currentOperation;
    private constructor();
    static getInstance(): ProgressManager;
    startOperation(operationId: string, total?: number): void;
    updateProgress(operationId: string, progress: number): void;
    completeOperation(operationId: string): void;
    getCurrentProgress(): {
        progress: number;
        total: number;
        elapsedTimeMs: number;
        estimatedTimeRemainingMs: number | null;
    };
    clearOperation(operationId: string): void;
}
