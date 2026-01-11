// frontend/js/app_v3.js

document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const taskGrid = document.getElementById('task-grid');
    const uploadModal = document.getElementById('upload-modal');
    const closeModal = document.getElementById('close-modal');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const processBtn = document.getElementById('process-btn');
    const selectedFilename = document.getElementById('selected-filename');
    const fileInfo = document.getElementById('file-info');
    const resultArea = document.getElementById('result-area');
    const resultJson = document.getElementById('result-json');
    const uploadPanel = document.getElementById('upload-panel');
    const processLoading = document.getElementById('process-loading');
    const resetBtn = document.getElementById('reset-btn');

    // --- State ---
    let currentTaskId = null;
    let selectedFile = null;
    let pollTimeoutId = null; // Para cancelar polling
    let isPolling = false;

    // --- Diagnose / UI Helpers ---
    const apiStatus = document.getElementById('api-status');
    const debugLogs = document.getElementById('debug-logs');

    function logMarker(label, data) {
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        const dataStr = (typeof data === 'object') ? JSON.stringify(data) : data;
        entry.innerHTML = `<span class="time">${timestamp}</span> <span class="label">${label}</span>: ${dataStr}`;
        if (debugLogs) debugLogs.prepend(entry);
        console.log(`[APP] ${label}:`, data || '');
    }

    // --- Initialization (Connection Check) ---
    async function initApp() {
        logMarker('INIT', `Backend URL: ${CONFIG.BACKEND_URL}`);

        try {
            // FIX: Usar /ping/ o /tasks/ en vez de / para asegurar JSON válido y evitar "Unexpected token <"
            const response = await fetch(`${CONFIG.BACKEND_URL}/tasks/`);

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            apiStatus.className = 'status-dot online';
            logMarker('CONNECTION', 'OK - Tareas cargadas');

            const mainLoading = document.getElementById('main-loading');
            if (mainLoading) mainLoading.remove();

            renderTasks(data.tasks);

        } catch (error) {
            apiStatus.className = 'status-dot offline';
            logMarker('INIT_ERROR', error.message);
            taskGrid.innerHTML = `
                <div style="text-align: center; color: var(--error); padding: 2rem;">
                    <h3><i class="bi bi-wifi-off"></i> Error de Conexión</h3>
                    <p>No se pudo conectar con el backend en: <code>${CONFIG.BACKEND_URL}</code></p>
                    <p class="text-muted">Verifica que el servidor Django esté corriendo y que la URL en config.js sea correcta.</p>
                    <button class="btn btn-secondary" onclick="location.reload()" style="margin-top:1rem">Reintentar</button>
                </div>
            `;
        }
    }

    function renderTasks(tasks) {
        taskGrid.innerHTML = ''; // Limpiar
        tasks.forEach(task => {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <h2><i class="bi ${task.icon || 'bi-gear'}"></i> ${task.name}</h2>
                <p>${task.description}</p>
                <div class="badge">${task.ext}</div>
            `;
            card.onclick = () => openUploadModal(task);
            taskGrid.appendChild(card);
        });
    }

    // --- Modal Logic ---
    function openUploadModal(task) {
        // Limpiamos estado anterior por seguridad
        stopPolling();

        currentTaskId = task.id;
        document.getElementById('modal-title').innerText = task.name;
        document.getElementById('modal-exts').innerText = `Formato: ${task.ext}`;

        resetUIForNewUpload();
        uploadModal.style.display = 'flex';
    }

    function resetUIForNewUpload() {
        uploadPanel.style.display = 'block';
        processLoading.style.display = 'none';
        resultArea.style.display = 'none';

        selectedFile = null;
        fileInput.value = ''; // Limpiar input file real
        selectedFilename.innerText = '';
        fileInfo.style.display = 'none';
        processBtn.disabled = true;
    }

    function closeModalHandler() {
        if (isPolling) {
            if (!confirm("Hay un análisis en curso. ¿Seguro que quieres cerrar? Se detendrá el monitoreo.")) return;
        }
        stopPolling();
        uploadModal.style.display = 'none';
    }

    closeModal.onclick = closeModalHandler;

    // Cerrar al dar click fuera (solo si no está procesando para evitar cierres accidentales)
    window.onclick = (event) => {
        if (event.target == uploadModal && !isPolling) {
            uploadModal.style.display = 'none';
        }
    };

    // --- File Drag & Drop ---
    dropZone.onclick = () => fileInput.click();
    fileInput.onchange = (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    };
    dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
    dropZone.ondragleave = () => dropZone.classList.remove('dragover');
    dropZone.ondrop = (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    };

    function handleFile(file) {
        selectedFile = file;
        selectedFilename.innerText = file.name;
        fileInfo.style.display = 'block';
        processBtn.disabled = false;
        logMarker('FILE_SELECTED', file.name);
    }

    // --- Processing Logic & Polling ---
    processBtn.onclick = async () => {
        if (!selectedFile || !currentTaskId) return;

        // UI Change
        uploadPanel.style.display = 'none';
        processLoading.style.display = 'block';
        document.getElementById('process-status').innerText = 'Subiendo archivo...';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const url = `${CONFIG.BACKEND_URL}/tasks/${currentTaskId}/`;

            logMarker('UPLOAD_START', url);
            const response = await fetch(url, { method: 'POST', body: formData });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.error || `Error ${response.status}`);
            }

            const data = await response.json();

            // Si es 202 Accepted, el backend está procesando en hilo aparte
            if (response.status === 202) {
                startPolling(data.task_id, data.file_id);
            } else {
                // Si respondiera directo (legacy)
                showResults(data.ml_result);
            }

        } catch (error) {
            handleError(error);
        }
    };

    function startPolling(taskId, fileId) {
        logMarker('POLLING_START', `Task: ${taskId}, File: ${fileId}`);
        document.getElementById('process-status').innerText = 'Procesando análisis (esto puede tardar)...';
        isPolling = true;

        let attempts = 0;
        const maxAttempts = 120; // ~4 minutos (intervalo 2s)

        const poll = async () => {
            if (!isPolling) return; // Cancelado

            attempts++;
            if (attempts > maxAttempts) {
                handleError(new Error("Timeout: El servidor tardó demasiado en responder."));
                return;
            }

            try {
                const res = await fetch(`${CONFIG.BACKEND_URL}/tasks/${taskId}/result/${fileId}/`);
                if (!res.ok) throw new Error("Error de red consultando estado");

                const data = await res.json();

                if (data.status === 'COMPLETED') {
                    logMarker('POLLING_DONE', 'Status COMPLETED');
                    isPolling = false;
                    showResults(data.ml_result);
                } else if (data.status === 'FAILED') {
                    throw new Error(JSON.stringify(data.error || "Falló el análisis en el servidor"));
                } else {
                    // Sigue procesando
                    logMarker('POLLING', `Intento ${attempts} - ${data.status}`);
                    pollTimeoutId = setTimeout(poll, 2000);
                }

            } catch (e) {
                // Errores de red transitorios no deberían matar el polling inmediatamente, 
                // pero si es persistente sí. Por ahora manejamos como error para simplificar.
                handleError(e);
            }
        };

        poll();
    }

    function stopPolling() {
        isPolling = false;
        if (pollTimeoutId) {
            clearTimeout(pollTimeoutId);
            pollTimeoutId = null;
        }
    }

    function handleError(error) {
        stopPolling();
        console.error(error);
        logMarker('ERROR', error.message);

        let msg = error.message;
        if (msg === 'Failed to fetch') msg = 'Error de conexión con el servidor.';

        alert(`Ocurrió un error: ${msg}`);

        // Regresar al estado de carga pero no cerrar modal completamente si es posible corregir
        processLoading.style.display = 'none';
        uploadPanel.style.display = 'block';
    }

    function showResults(data) {
        processLoading.style.display = 'none';
        resultArea.style.display = 'block';

        // Construir UI de resultados
        // Header
        let html = `<div class="analysis-summary">
            <h3><i class="bi bi-check-circle-fill" style="color:var(--success)"></i> Análisis Completado</h3>
            <p>${data.result || 'Resultados generados exitosamente.'}</p>
        </div>`;

        // Métricas
        if (data.metrics && data.metrics.length > 0) {
            html += `<div style="display:flex; flex-wrap:wrap; gap:10px; margin-bottom:2rem;">`;
            data.metrics.forEach(m => {
                html += `<div class="badge">${m.label}: ${m.value}</div>`;
            });
            html += `</div>`;
        }

        // Tablas
        if (data.tables) {
            data.tables.forEach(t => {
                html += `<div class="table-viewer">
                    <h4>${t.title}</h4>
                    <div class="table-container">${t.content}</div>
                </div>`;
            });
        }

        // Gráficos
        if (data.graphics) {
            html += `<div class="graphics-grid">`;
            data.graphics.forEach(g => {
                html += `<div class="graphic-item">
                    <h4>${g.title}</h4>
                    <img src="data:image/png;base64,${g.image}" loading="lazy" alt="${g.title}">
                </div>`;
            });
            html += `</div>`;
        }

        // Botón Reset
        html += `<div style="margin-top:2rem; text-align:center;">
            <button class="btn btn-primary" id="new-analysis-btn">Nuevo Análisis</button>
        </div>`;

        resultArea.innerHTML = html;

        // Bind reset button dinámicamente
        document.getElementById('new-analysis-btn').onclick = () => {
            resetUIForNewUpload();
        };
    }

    // --- Manual Helpers ---
    resetBtn.onclick = resetUIForNewUpload;
    document.getElementById('test-conn-btn').onclick = async () => {
        try {
            const res = await fetch(`${CONFIG.BACKEND_URL}/ping/`, { method: 'POST' });
            const d = await res.json();
            alert(`Conexión Exitosa: ${d.status}`);
        } catch (e) {
            alert(`Error de Conexión: ${e.message}\nVerifica que el backend corra en ${CONFIG.BACKEND_URL}`);
        }
    };

    // Start
    initApp();
});
