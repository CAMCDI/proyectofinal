// js/state.js - Estado global de la aplicación
const AppState = {
    currentTask: null,
    currentFileId: null,

    setTask(task) {
        this.currentTask = task;
        localStorage.setItem('currentTask', JSON.stringify(task));
    },

    getTask() {
        if (!this.currentTask) {
            const stored = localStorage.getItem('currentTask');
            this.currentTask = stored ? JSON.parse(stored) : null;
        }
        return this.currentTask;
    },

    setFileId(fileId) {
        this.currentFileId = fileId;
        localStorage.setItem('currentFileId', fileId);
    },

    getFileId() {
        if (!this.currentFileId) {
            this.currentFileId = localStorage.getItem('currentFileId');
        }
        return this.currentFileId;
    },

    clear() {
        this.currentTask = null;
        this.currentFileId = null;
        localStorage.removeItem('currentTask');
        localStorage.removeItem('currentFileId');
    }
};
