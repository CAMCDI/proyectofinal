// file_manager/static/file_manager/js/ml_actions.js

document.addEventListener('DOMContentLoaded', function () {
    const processButtons = document.querySelectorAll('.process-btn');
    const resultPanel = document.getElementById('ml-result-panel');
    const resultTask = document.getElementById('ml-result-task');
    const resultText = document.getElementById('ml-result-text');
    const resultDetails = document.getElementById('ml-result-details');
    const loadingOverlay = document.getElementById('ml-loading-overlay');

    processButtons.forEach(button => {
        button.addEventListener('click', function () {
            const action = this.dataset.action;
            const fileId = this.dataset.fileId;
            const processUrl = this.dataset.url;

            // Show loading
            loadingOverlay.classList.remove('d-none');
            resultPanel.classList.add('d-none');

            const formData = new FormData();
            formData.append('action', action);

            // Get CSRF token from cookie
            const csrftoken = getCookie('csrftoken');

            fetch(processUrl, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': csrftoken,
                }
            })
                .then(response => response.json())
                .then(data => {
                    loadingOverlay.classList.add('d-none');

                    if (data.success) {
                        resultTask.textContent = data.task;
                        resultText.textContent = data.result;
                        resultDetails.textContent = data.details;

                        // Style based on result
                        resultText.className = 'fw-bold';
                        if (data.result.includes('SPAM') || data.result.includes('Anomaly')) {
                            resultText.classList.add('text-danger');
                        } else {
                            resultText.classList.add('text-success');
                        }

                        resultPanel.classList.remove('d-none');
                        resultPanel.scrollIntoView({ behavior: 'smooth' });
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    loadingOverlay.classList.add('d-none');
                    console.error('Error:', error);
                    alert('Ocurrió un error al procesar el archivo.');
                });
        });
    });

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});
