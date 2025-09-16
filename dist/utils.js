export function sanitizeProjectName(projectName) {
    return projectName.replace(/[^a-zA-Z0-9]/g, "_");
}
export function capitalizeFirstLetter(val) {
    return String(val).charAt(0).toUpperCase() + String(val).slice(1);
}
//# sourceMappingURL=utils.js.map