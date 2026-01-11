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
            card.innerHTML = `
                <h2><i class="bi ${task.icon || 'bi-gear'}"></i> ${task.name}</h2>
                <p>${task.description}</p>
                <div class="badge">${task.ext}</div>
            `;
            card.onclick = () => {
                AppState.setTask(task);
                window.location.href = 'upload.html';
            };
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
