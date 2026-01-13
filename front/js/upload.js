document.addEventListener('DOMContentLoaded', () => {
    const task = AppState.getTask();
    if (!task) {
        window.location.href = 'index.html';
        return;
    }

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const selectedFilename = document.getElementById('selected-filename');
    const fileInfo = document.getElementById('file-info');
    const uploadPanel = document.getElementById('upload-panel');
    const uploadLoading = document.getElementById('upload-loading');
    const uploadStatus = document.getElementById('upload-status');

    // Inicializar info de la tarea
    document.getElementById('task-title').innerText = task.name;
    document.getElementById('task-description').innerText = task.description;
    document.getElementById('task-ext').innerText = `Formato: ${task.ext}`;

    let selectedFile = null;

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
        uploadBtn.disabled = false;
    }

    // Proceso de subida
    uploadBtn.onclick = async () => {
        if (!selectedFile) return;

        const progressBar = document.getElementById('upload-progress');
        uploadPanel.style.display = 'none';
        uploadLoading.style.display = 'block';
        uploadStatus.innerText = 'Initializing Data Injection... (0%)';
        progressBar.style.width = '0%';

        try {
            const data = await API.uploadFile(task.id, selectedFile, (percent) => {
                progressBar.style.width = `${percent}%`;
                uploadStatus.innerText = `Injecting Data Packets... (${percent}%)`;
                if (percent === 100) {
                    uploadStatus.innerText = 'Analyzing Patterns in Mainframe...';
                }
            });

            if (data.status === 'accepted' && data.file_id) {
                AppState.setFileId(data.file_id);
                window.location.href = 'result.html';
            } else {
                throw new Error('Respuesta inesperada del servidor');
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
            uploadPanel.style.display = 'block';
            uploadLoading.style.display = 'none';
        }
    };
});
