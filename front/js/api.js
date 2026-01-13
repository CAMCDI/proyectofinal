const API = {
    async fetchWithRetry(url, options = {}, retries = 3, backoff = 1000) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.error || `Error HTTP ${response.status}`);
            }
            return response.json();
        } catch (error) {
            // Reintentar solo en errores de red
            if (retries > 0 && (error.message.includes('Failed to fetch') || error.message.includes('NetworkError'))) {
                console.warn(`Reintentando ${url}... Intentos restantes: ${retries}`);
                await new Promise(r => setTimeout(r, backoff));
                return this.fetchWithRetry(url, options, retries - 1, backoff * 1.5);
            }
            throw error;
        }
    },

    async getTasks() {
        return this.fetchWithRetry(`${CONFIG.BACKEND_URL}/tasks/`);
    },

    async uploadFile(taskId, file) {
        const formData = new FormData();
        formData.append('file', file);
        return this.fetchWithRetry(`${CONFIG.BACKEND_URL}/tasks/${taskId}/`, {
            method: 'POST',
            body: formData
        }, 1);
    },

    async getResult(taskId, fileId) {
        const response = await fetch(`${CONFIG.BACKEND_URL}/tasks/${taskId}/result/${fileId}/`);
        if (!response.ok) throw new Error('Error al obtener resultados');
        return response.json();
    },

    async pollResult(taskId, fileId, onProgress) {
        let attempts = 0;
        const maxAttempts = 120;
        await new Promise(r => setTimeout(r, 1000));

        return new Promise((resolve, reject) => {
            const poll = async () => {
                if (attempts++ > maxAttempts) {
                    reject(new Error('El análisis tardó demasiado tiempo'));
                    return;
                }

                try {
                    const data = await this.getResult(taskId, fileId);
                    if (data.status === 'COMPLETED') {
                        resolve(data.ml_result);
                    } else if (data.status === 'FAILED') {
                        let errorMsg = data.error?.error || data.error || 'Error desconocido';
                        reject(new Error(errorMsg));
                    } else {
                        if (onProgress) onProgress(data.status, attempts);
                        setTimeout(poll, 2000);
                    }
                } catch (error) {
                    if (attempts < maxAttempts) {
                        if (onProgress) onProgress('REINTENTANDO', attempts);
                        setTimeout(poll, 2000);
                    } else {
                        reject(error.message.includes('Failed to fetch') ? new Error("Error de conexión.") : error);
                    }
                }
            };
            poll();
        });
    }
};
