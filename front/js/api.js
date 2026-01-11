// js/api.js - API client para comunicación con backend
const API = {
    async fetchWithRetry(url, options = {}, retries = 3, backoff = 1000) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                // If 500 or 503, maybe worth retrying? For now only retry network errors
                // But if response came back, it's not a network error.
                // Throw to handle application error
                const error = await response.json().catch(() => ({}));
                throw new Error(error.error || `HTTP Error ${response.status}`);
            }
            return response.json();
        } catch (error) {
            if (retries > 0 && (error.message.includes('Failed to fetch') || error.message.includes('NetworkError'))) {
                console.warn(`Retrying ${url}... Attempts left: ${retries}`);
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
        }, 1); // Retry upload only once to avoid duplicate huge uploads
    },

    async getResult(taskId, fileId) {
        const response = await fetch(`${CONFIG.BACKEND_URL}/tasks/${taskId}/result/${fileId}/`);
        if (!response.ok) throw new Error('Error fetching result');
        return response.json();
    },

    async pollResult(taskId, fileId, onProgress) {
        let attempts = 0;
        const maxAttempts = 120;

        // Initial delay to allow backend to initialize thread/db
        await new Promise(r => setTimeout(r, 1000));

        return new Promise((resolve, reject) => {
            const poll = async () => {
                if (attempts++ > maxAttempts) {
                    reject(new Error('Timeout: análisis tardó demasiado'));
                    return;
                }

                try {
                    const data = await this.getResult(taskId, fileId);

                    if (data.status === 'COMPLETED') {
                        resolve(data.ml_result);
                    } else if (data.status === 'FAILED') {
                        // Extract specific error message if possible
                        let errorMsg = 'Error desconocido en análisis';
                        if (data.error) {
                            if (typeof data.error === 'string') {
                                errorMsg = data.error;
                            } else if (data.error.error) {
                                errorMsg = data.error.error;
                            } else {
                                errorMsg = JSON.stringify(data.error);
                            }
                        }
                        reject(new Error(errorMsg));
                    } else {
                        if (onProgress) onProgress(data.status, attempts);
                        setTimeout(poll, 2000);
                    }
                } catch (error) {
                    console.warn(`Intento ${attempts} fallido:`, error);

                    // RETRY LOGIC: If network error or 500, keep trying until timeout
                    if (attempts < maxAttempts) {
                        if (onProgress) onProgress('RETRYING', attempts);
                        setTimeout(poll, 2000);
                    } else {
                        // Final rejection
                        if (error.message.includes('Failed to fetch')) {
                            reject(new Error("Error de conexión con el servidor (Failed to fetch)."));
                        } else {
                            reject(error);
                        }
                    }
                }
            };

            poll();
        });
    }
};
