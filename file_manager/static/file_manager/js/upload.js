/* file_manager/static/file_manager/js/upload.js */

document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const browseBtn = document.getElementById('browseBtn');
    const filePreview = document.getElementById('filePreview');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeFile = document.getElementById('removeFile');
    const submitBtn = document.getElementById('submitBtn');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    // Configuration from JSON scripts
    let allowedExts = [];
    let maxSizeMB = 10;

    try {
        const extData = document.getElementById('allowed-extensions');
        if (extData) allowedExts = JSON.parse(extData.textContent);

        const sizeData = document.getElementById('max-size-mb');
        if (sizeData) maxSizeMB = JSON.parse(sizeData.textContent);
    } catch (e) {
        console.error('Error loading configuration:', e);
    }

    // Browse button click -> trigger hidden file input
    if (browseBtn && fileInput) {
        browseBtn.addEventListener('click', () => fileInput.click());
    }

    // File input change
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            if (this.files && this.files.length > 0) {
                handleFile(this.files[0]);
            }
        });
    }

    // Drag and Drop
    if (dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('bg-light', 'border-primary'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('bg-light', 'border-primary'), false);
        });

        dropZone.addEventListener('drop', handleDrop, false);
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            fileInput.files = files; // Note: This works in modern browsers
            handleFile(files[0]);
        }
    }

    function handleFile(file) {
        // Reset errors
        if (errorMessage) errorMessage.classList.add('d-none');

        const name = file.name;
        const ext = name.includes('.') ? name.split('.').pop().toLowerCase() : '';
        const size = file.size / (1024 * 1024);

        // Validation
        if (!allowedExts.includes(ext)) {
            showError(`Extensión .${ext} no permitida. Use: ${allowedExts.join(', ')}`);
            resetSelection();
            return;
        }

        if (size > maxSizeMB) {
            showError(`Archivo demasiado grande (${size.toFixed(2)} MB). Máximo ${maxSizeMB} MB.`);
            resetSelection();
            return;
        }

        // Update UI
        if (fileName) fileName.textContent = name;
        if (fileSize) fileSize.textContent = formatBytes(file.size);

        if (dropZone) dropZone.classList.add('d-none');
        if (filePreview) filePreview.classList.remove('d-none');
        if (submitBtn) submitBtn.disabled = false;
    }

    function showError(msg) {
        if (errorText) errorText.textContent = msg;
        if (errorMessage) errorMessage.classList.remove('d-none');
    }

    function resetSelection() {
        if (fileInput) fileInput.value = '';
        if (dropZone) dropZone.classList.remove('d-none');
        if (filePreview) filePreview.classList.add('d-none');
        if (submitBtn) submitBtn.disabled = true;
    }

    if (removeFile) {
        removeFile.addEventListener('click', resetSelection);
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
});
