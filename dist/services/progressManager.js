/**
 * A simple manager to track progress of various operations
 */
const FUDGE_FACTOR = 1.2; // seat of my pants guess
export class ProgressManager {
    static instance;
    progressMap = new Map();
    currentOperation = null;
    constructor() { }
    static getInstance() {
        if (!ProgressManager.instance) {
            ProgressManager.instance = new ProgressManager();
        }
        return ProgressManager.instance;
    }
    startOperation(operationId, total = 100) {
        const now = Date.now();
        this.progressMap.set(operationId, { progress: 0, total, startTime: now, lastUpdateTime: now });
        this.currentOperation = operationId;
    }
    updateProgress(operationId, progress) {
        const currentProgress = this.progressMap.get(operationId);
        if (currentProgress) {
            this.progressMap.set(operationId, {
                progress,
                total: currentProgress.total,
                startTime: currentProgress.startTime,
                lastUpdateTime: Date.now()
            });
        }
    }
    completeOperation(operationId) {
        const currentProgress = this.progressMap.get(operationId);
        if (currentProgress) {
            this.progressMap.set(operationId, {
                progress: currentProgress.total,
                total: currentProgress.total,
                startTime: currentProgress.startTime,
                lastUpdateTime: Date.now()
            });
        }
        if (this.currentOperation === operationId) {
            this.currentOperation = null;
        }
    }
    getCurrentProgress() {
        if (this.currentOperation) {
            const operationData = this.progressMap.get(this.currentOperation);
            if (operationData) {
                const now = Date.now();
                const elapsedTimeMs = now - operationData.startTime;
                let estimatedTimeRemainingMs = null;
                // Only estimate if we have meaningful progress (at least 5% complete and some time elapsed)
                if (operationData.progress > 0.05 * operationData.total && elapsedTimeMs > 1000) {
                    const progressPercentage = operationData.progress / operationData.total;
                    const estimatedTotalTime = (elapsedTimeMs / progressPercentage) * FUDGE_FACTOR;
                    estimatedTimeRemainingMs = Math.max(0, estimatedTotalTime - elapsedTimeMs);
                }
                return {
                    progress: operationData.progress,
                    total: operationData.total,
                    elapsedTimeMs,
                    estimatedTimeRemainingMs
                };
            }
        }
        return { progress: 0, total: 100, elapsedTimeMs: 0, estimatedTimeRemainingMs: null };
    }
    clearOperation(operationId) {
        this.progressMap.delete(operationId);
        if (this.currentOperation === operationId) {
            this.currentOperation = null;
        }
    }
}
//# sourceMappingURL=progressManager.js.map