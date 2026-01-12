// frontend/js/app_v3.js

// frontend/js/app_v3.js
/**
 * Main Client Logic
 * Refactored for Clean Code & Maintainability
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration & State ---
    const State = {
        taskId: null,
        file: null,
        isPolling: false,
        pollTimeout: null
    };

    // --- UI Strings & Renderers (Separation of Concerns) ---
    const UIManager = {
        els: {
            taskGrid: document.getElementById('task-grid'),
            modal: document.getElementById('upload-modal'),
            uploadPanel: document.getElementById('upload-panel'),
            processLoading: document.getElementById('process-loading'),
            resultArea: document.getElementById('result-area'),
            fileInput: document.getElementById('file-input'),
            fileInfo: document.getElementById('file-info'),
            filename: document.getElementById('selected-filename'),
            processBtn: document.getElementById('process-btn'),
            statusDot: document.getElementById('api-status'),
            debugLogs: document.getElementById('debug-logs')
        },

        renderTasks(tasks) {
            this.els.taskGrid.innerHTML = tasks.map(task => `
                <div class="card" onclick="startTaskSelection('${task.id}', '${task.name}', '${task.ext}')">
                    <h2><i class="bi ${task.icon || 'bi-gear'}"></i> ${task.name}</h2>
                    <p>${task.description}</p>
                    <div class="badge">${task.ext}</div>
                </div>
            `).join('');

            // Re-attach handlers needing closure data if necessary, 
            // but inline onclick is used here for simplicity vs boilerplate
        },

        showModal(title, ext) {
            document.getElementById('modal-title').innerText = title;
            document.getElementById('modal-exts').innerText = `Formato: ${ext}`;
            this.toggleView('upload');
            this.els.modal.style.display = 'flex';
        },

        closeModal() {
            this.els.modal.style.display = 'none';
        },

        toggleView(view) {
            // Simple State Machine for Modal UI
            this.els.uploadPanel.style.display = (view === 'upload') ? 'block' : 'none';
            this.els.processLoading.style.display = (view === 'loading') ? 'block' : 'none';
            this.els.resultArea.style.display = (view === 'result') ? 'block' : 'none';
        },

        updateStatus(isOnline) {
            this.els.statusDot.className = isOnline ? 'status-dot online' : 'status-dot offline';
        },

        fileSelected(file) {
            this.els.filename.innerText = file.name;
            this.els.fileInfo.style.display = 'block';
            this.els.processBtn.disabled = false;
        },

        resetUpload() {
            this.els.fileInput.value = '';
            this.els.filename.innerText = '';
            this.els.fileInfo.style.display = 'none';
            this.els.processBtn.disabled = true;
            this.toggleView('upload');
        },

        renderResults(data) {
            this.toggleView('result');

            const metricsHtml = (data.metrics || []).map(m =>
                `<div class="badge" style="font-size:0.9em">${m.label}: ${m.value}</div>`
            ).join('');

            const tablesHtml = (data.tables || []).map(t => `
                <div class="table-viewer">
                    <h4 style="margin-bottom:10px; color:var(--secondary)">${t.title}</h4>
                    <div class="table-container">${t.content}</div>
                </div>
            `).join('');

            const graphicsHtml = (data.graphics || []).map(g => `
                <div class="graphic-item">
                    <h4>${g.title}</h4>
                    <img src="data:image/png;base64,${g.image}" loading="lazy" alt="${g.title}">
                </div>
            `).join('');

            this.els.resultArea.innerHTML = `
                <div class="analysis-summary">
                    <h3 style="margin-top:0"><i class="bi bi-check-circle-fill"></i> Análisis Completado</h3>
                    <p>${data.result || 'Proceso finalizado con éxito.'}</p>
                </div>
                
                <div style="display:flex; gap:10px; margin-bottom:2rem; flex-wrap:wrap">
                    ${metricsHtml}
                </div>

                ${tablesHtml}
                
                <div class="graphics-grid">
                    ${graphicsHtml}
                </div>

                <div style="margin-top:3rem; text-align:center;">
                    <button class="btn btn-primary" onclick="resetApp()">
                        <i class="bi bi-arrow-counterclockwise"></i> Nuevo Análisis
                    </button>
                </div>
            `;
        }
    };

    // --- Logic / Controller ---

    const Logger = {
        log(label, data) {
            const time = new Date().toLocaleTimeString();
            const msg = (typeof data === 'object') ? JSON.stringify(data) : data;
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="time">${time}</span> <span class="label">${label}</span> ${msg}`;

            const logs = document.getElementById('debug-logs');
            if (logs) logs.prepend(entry);
            console.log(`[APP] ${label}:`, data);
        }
    };

    async function init() {
        Logger.log('INIT', `Connecting to ${CONFIG.BACKEND_URL}`);
        try {
            const res = await API.getTasks(); // Assumes API.getTasks logic from api.js
            UIManager.updateStatus(true);
            UIManager.renderTasks(res.tasks);
            Logger.log('READY', `${res.tasks.length} tasks loaded`);

            // Remove initial skeleton/loading
            const loader = document.querySelector('.loading');
            if (loader) loader.style.display = 'none';

        } catch (err) {
            UIManager.updateStatus(false);
            Logger.log('ERROR', err.message);
            // Optional: Show error UI in grid
            alert('Error conectando al backend. Ver consola.');
        }
    }

    // --- Actions ---

    window.startTaskSelection = (id, name, ext) => {
        stopActivePolling();
        State.taskId = id;
        UIManager.showModal(name, ext);
        UIManager.resetUpload();
        State.file = null;
    };

    window.resetApp = () => {
        UIManager.resetUpload();
        State.file = null;
    };

    // File Handling
    const handleFile = (file) => {
        if (!file) return;
        State.file = file;
        UIManager.fileSelected(file);
    };

    UIManager.els.fileInput.onchange = (e) => handleFile(e.target.files[0]);

    // Drag & Drop
    const dz = document.getElementById('drop-zone');
    dz.onclick = () => UIManager.els.fileInput.click();
    dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('dragover'); };
    dz.ondragleave = () => dz.classList.remove('dragover');
    dz.ondrop = (e) => {
        e.preventDefault();
        dz.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0]);
    };

    // Process Flow
    UIManager.els.processBtn.onclick = async () => {
        if (!State.file || !State.taskId) return;

        UIManager.toggleView('loading');
        document.getElementById('process-status').innerText = 'Subiendo archivo...';

        try {
            const res = await API.uploadFile(State.taskId, State.file);
            Logger.log('UPLOAD', 'Success');

            if (res.status === 'accepted' || res.status === 'success') { // Handle both
                initPolling(State.taskId, res.file_id || res.task_id); // Adjust based on API return
            } else {
                UIManager.renderResults(res.ml_result); // Direct result
            }
        } catch (err) {
            handleError(err);
        }
    };

    // Polling Logic
    function initPolling(taskId, fileId) {
        State.isPolling = true;
        document.getElementById('process-status').innerText = 'Procesando... esto puede tardar unos segundos.';

        API.pollResult(taskId, fileId, (status, attempt) => {
            if (!State.isPolling) return;
            Logger.log('POLL', `Attempt ${attempt} - ${status}`);
        })
            .then(result => {
                if (!State.isPolling) return;
                Logger.log('DONE', 'Analysis complete');
                UIManager.renderResults(result);
                State.isPolling = false;
            })
            .catch(handleError);
    }

    function stopActivePolling() {
        State.isPolling = false;
        if (State.pollTimeout) clearTimeout(State.pollTimeout);
    }

    function handleError(err) {
        State.isPolling = false;
        Logger.log('ERROR', err.message);
        alert(`Error: ${err.message}`);
        UIManager.toggleView('upload'); // Go back so user can retry
    }

    // Global Modal Close Logic
    const closeBtn = document.getElementById('close-modal');
    closeBtn.onclick = () => {
        if (State.isPolling && !confirm("¿Cancelar análisis?")) return;
        stopActivePolling();
        UIManager.closeModal();
    };

    // Initialize
    init();
});
