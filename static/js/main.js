document.addEventListener('DOMContentLoaded', () => {
    const el = (id) => document.getElementById(id);
    let currentFile = null;

    // File Selection
    el('drop-zone').onclick = (e) => {
        if (e.target.tagName !== 'LABEL' && e.target.tagName !== 'INPUT') {
            el('file-input').click();
        }
    };
    el('file-input').onchange = (e) => handleFile(e.target.files[0]);

    // Drag and Drop Events
    el('drop-zone').ondragover = (e) => { e.preventDefault(); el('drop-zone').classList.add('dragover'); };
    el('drop-zone').ondragleave = (e) => { e.preventDefault(); el('drop-zone').classList.remove('dragover'); };
    el('drop-zone').ondrop = (e) => {
        e.preventDefault();
        el('drop-zone').classList.remove('dragover');
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    };

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/') || file.size > 16 * 1024 * 1024) {
            return alert('Please upload a valid image file (Max 16MB).');
        }
        
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            el('input-preview').src = e.target.result;
            el('upload-section').classList.add('hidden');
            el('preview-section').classList.remove('hidden');
            
            // Reset Output UI
            el('output-preview').src = '';
            el('output-preview').classList.add('hidden');
            el('output-container').classList.add('skeleton-container');
            el('stylize-btn').classList.remove('hidden');
            el('download-btn').classList.add('hidden');
            el('processing-overlay').classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    el('reset-btn').onclick = () => {
        currentFile = null;
        el('file-input').value = '';
        el('preview-section').classList.add('hidden');
        el('upload-section').classList.remove('hidden');
    };

    el('stylize-btn').onclick = async () => {
        if (!currentFile) return;

        el('stylize-btn').classList.add('hidden');
        el('download-btn').classList.add('hidden');
        el('output-preview').classList.add('hidden');
        el('output-container').classList.add('skeleton-container');
        el('processing-overlay').classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('model', el('model-select').value);

        try {
            const res = await fetch('/stylize', { method: 'POST', body: formData });
            const data = await res.json();

            if (res.ok && data.success) {
                el('output-preview').src = data.output_url;
                el('output-preview').onload = () => {
                    el('processing-overlay').classList.add('hidden');
                    el('output-container').classList.remove('skeleton-container');
                    el('output-preview').classList.remove('hidden');
                    el('download-btn').href = data.output_url;
                    el('download-btn').classList.remove('hidden');
                    el('stylize-btn').classList.remove('hidden');
                };
            } else throw new Error(data.error || 'Failed to process image');
        } catch (err) {
            alert('Error: ' + err.message);
            el('processing-overlay').classList.add('hidden');
            el('stylize-btn').classList.remove('hidden');
        }
    };
});
