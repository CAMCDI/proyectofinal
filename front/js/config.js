// frontend/js/config.js
const CONFIG = {
    // Detecta automáticamente si estamos en local o prod
    BACKEND_URL: (function () {
        const hostname = window.location.hostname;
        // Si estamos en localhost o 127.0.0.1, usamos el backend local
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://127.0.0.1:8000';
        }
        // Si no, asumimos producción/túnel. 
        // IMPORTANTE: El usuario debe poner aquí su URL fija de Tunnelto si no quiere depender de lógica automática.
        // O dejar esta lógica para que funcione si sirven el front desde el mismo dominio (no es el caso aquí).

        // Poner aquí la URL de tu túnel (ej: https://mi-app.tunnelto.dev)
        return 'https://t-fxzfd59t.tunn.dev';
    })()
};

console.log('[CONFIG] Detected Backend URL:', CONFIG.BACKEND_URL);
