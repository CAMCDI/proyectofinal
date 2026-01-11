// js/result.js - Lógica de polling y renderizado de resultados
document.addEventListener('DOMContentLoaded', async () => {
    const task = AppState.getTask();
    const fileId = AppState.getFileId();

    if (!task || !fileId) {
        window.location.href = 'index.html';
        return;
    }

    // UI Elements
    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');
    const errorSection = document.getElementById('error-section');
    const progressText = document.getElementById('progress-text');
    const resultContent = document.getElementById('result-content');

    // Set task info
    document.getElementById('task-title').innerText = task.name;
    document.getElementById('task-description').innerText = task.description;

    try {
        const result = await API.pollResult(task.id, fileId, (status, attempts) => {
            progressText.innerText = `Estado: ${status} (intento ${attempts})`;
        });

        loadingSection.style.display = 'none';
        resultSection.style.display = 'block';

        renderResult(result);

    } catch (error) {
        loadingSection.style.display = 'none';
        errorSection.style.display = 'block';
        document.getElementById('error-message').innerText = error.message;
    }

    function renderResult(data) {
        document.getElementById('result-summary').innerText = data.result || 'Análisis completado exitosamente';

        let html = '';

        // Métricas
        if (data.metrics && data.metrics.length > 0) {
            html += '<div style="display:flex; flex-wrap:wrap; gap:10px; margin-bottom:2rem;">';
            data.metrics.forEach(m => {
                html += `<div class="badge">${m.label}: ${m.value}</div>`;
            });
            html += '</div>';
        }

        // Tablas
        if (data.tables) {
            data.tables.forEach(t => {
                html += `
                    <div class="table-viewer">
                        <h4>${t.title}</h4>
                        <div class="table-container">${t.content}</div>
                    </div>
                `;
            });
        }

        // Gráficos
        if (data.graphics) {
            html += '<div class="graphics-grid">';
            data.graphics.forEach(g => {
                html += `
                    <div class="graphic-item">
                        <h4>${g.title}</h4>
                        <img src="data:image/png;base64,${g.image}" loading="lazy" alt="${g.title}">
                    </div>
                `;
            });
            html += '</div>';
        }

        resultContent.innerHTML = html;
    }
});
