// js/index.js - Lógica de selección de ejercicios
document.addEventListener('DOMContentLoaded', async () => {
    const taskGrid = document.getElementById('task-grid');
    const apiStatus = document.getElementById('api-status');

    try {
        const data = await API.getTasks();
        apiStatus.className = 'status-dot online';

        taskGrid.innerHTML = '';
        data.tasks.forEach(task => {
            const card = document.createElement('div');
            card.className = 'card';
            card.onclick = () => {
                AppState.setTask(task);
                window.location.href = 'upload.html';
            };

            card.innerHTML = `
                <div class="card-header">
                    <div class="card-icon">
                        <i class="bi ${task.icon}"></i>
                    </div>
                    <h2>${task.name}</h2>
                </div>
                <p>${task.description}</p>
                <div class="card-footer">
                    <span class="badge">${task.ext}</span>
                    <i class="bi bi-chevron-right" style="color: var(--neon-cyan)"></i>
                </div>
            `;
            taskGrid.appendChild(card);
        });
    } catch (error) {
        apiStatus.className = 'status-dot offline';
        taskGrid.innerHTML = `
            <div style="text-align: center; color: var(--error); padding: 2rem;">
                <h3><i class="bi bi-wifi-off"></i> Error de Conexión</h3>
                <p>No se pudo conectar al backend: ${CONFIG.BACKEND_URL}</p>
                <button class="btn btn-secondary" onclick="location.reload()">Reintentar</button>
            </div>
        `;
    }
});
