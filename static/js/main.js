document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const previewSection = document.getElementById('preview-section');
    const inputPreview = document.getElementById('input-preview');
    const outputPreview = document.getElementById('output-preview');
    const stylizeBtn = document.getElementById('stylize-btn');
    const resetBtn = document.getElementById('reset-btn');
    const downloadBtn = document.getElementById('download-btn');
    const processingOverlay = document.getElementById('processing-overlay');
    const outputContainer = document.getElementById('output-container');

    let currentFile = null;

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    
    // Make entire dropzone clickable
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Prevent double triggering when clicking the label/button inside
    document.querySelector('label[for="file-input"]').addEventListener('click', (e) => {
        e.stopPropagation();
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length) {
            handleFile(files[0]);
        }
    }

    function handleFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid image file (JPG or PNG).');
            return;
        }

        // Validate file size (max 16MB)
        if (file.size > 16 * 1024 * 1024) {
            alert('File size exceeds 16MB limit.');
            return;
        }

        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            inputPreview.src = e.target.result;
            
            // Switch views
            uploadSection.classList.add('hidden');
            previewSection.classList.remove('hidden');
            
            // Reset output state
            outputPreview.src = '';
            outputPreview.classList.add('hidden');
            outputContainer.classList.add('skeleton-container');
            stylizeBtn.classList.remove('hidden');
            downloadBtn.classList.add('hidden');
            processingOverlay.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        previewSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    });

    stylizeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Update UI to processing state
        stylizeBtn.classList.add('hidden');
        processingOverlay.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', currentFile);
        
        const modelSelect = document.getElementById('model-select');
        if (modelSelect) {
            formData.append('model', modelSelect.value);
        }

        try {
            const response = await fetch('/stylize', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Success
                outputPreview.src = data.output_url;
                
                // Once image is loaded, update UI
                outputPreview.onload = () => {
                    processingOverlay.classList.add('hidden');
                    outputContainer.classList.remove('skeleton-container');
                    outputPreview.classList.remove('hidden');
                    
                    downloadBtn.href = data.output_url;
                    downloadBtn.classList.remove('hidden');
                };
            } else {
                // Handle server error
                throw new Error(data.error || 'Failed to process image');
            }
        } catch (error) {
            alert('Error: ' + error.message);
            
            // Reset UI on error
            processingOverlay.classList.add('hidden');
            stylizeBtn.classList.remove('hidden');
        }
    });
});
