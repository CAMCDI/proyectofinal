document.addEventListener('DOMContentLoaded', async () => {
    const task = AppState.getTask();
    const fileId = AppState.getFileId();

    if (!task || !fileId) {
        window.location.href = 'index.html';
        return;
    }

    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');
    const errorSection = document.getElementById('error-section');
    const progressText = document.getElementById('progress-text');
    const resultContent = document.getElementById('result-content');

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

    function renderResults(result) {
        resultSection.style.display = 'block';
        loadingSection.style.display = 'none';

        document.getElementById('result-summary').innerText = result.result;

        let html = '';

        // 1. Métricas como tarjetas de dashboard
        if (result.metrics && result.metrics.length > 0) {
            html += '<div class="metrics-grid">';
            result.metrics.forEach(m => {
                html += `
                    <div class="metric-card">
                        <span class="label">${m.label}</span>
                        <span class="value">${m.value}</span>
                    </div>
                `;
            });
            html += '</div>';
        }

        // 2. Tablas de datos
        if (result.tables && result.tables.length > 0) {
            result.tables.forEach(table => {
                html += `
                    <div class="table-viewer">
                        <h4 class="section-title">${table.title}</h4>
                        <div class="table-wrapper">
                            ${table.content}
                        </div>
                    </div>
                `;
            });
        }

        // 3. Gráficas
        if (result.graphics && result.graphics.length > 0) {
            html += '<h4 class="section-title">Visualización de Inteligencia</h4>';
            html += '<div class="graphics-container">';
            result.graphics.forEach(g => {
                html += `
                    <div class="graphic-box">
                        <h4>${g.title}</h4>
                        <img src="data:image/png;base64,${g.image}" alt="${g.title}">
                    </div>
                `;
            });
            html += '</div>';
        }

        resultContent.innerHTML = html;
    }
});
