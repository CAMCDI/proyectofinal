// frontend/js/config.js
const CONFIG = {
    // Detecta automáticamente si estamos en local o prod
    BACKEND_URL: (function () {
        const hostname = window.location.hostname;
        // Entorno de desarrollo local: API2 corre en el puerto 8001
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://127.0.0.1:8001';
        }

        // Entorno de producción (Render):
        // Aquí debes poner la URL pública de tu API2 desplegada en Render.
        return 'https://api-gateway-render-url.onrender.com';
    })()
};

console.log('[CONFIG] Detected Backend URL:', CONFIG.BACKEND_URL);
