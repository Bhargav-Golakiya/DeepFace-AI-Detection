const UI = {
    loading: document.getElementById('loading-overlay')
};

// Helper: Wire up drag-and-drop and file input selection for an upload box
function setupUpload(uploadAreaId, fileInputId, previewImgId, callback) {
    const area = document.getElementById(uploadAreaId);
    const input = document.getElementById(fileInputId);
    const preview = document.getElementById(previewImgId);
    if (!area || !input) return;

    area.addEventListener('click', () => input.click());
    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('dragover');
    });
    area.addEventListener('dragleave', () => area.classList.remove('dragover'));
    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleFile(input.files[0], preview, area, callback);
        }
    });
    input.addEventListener('change', () => {
        if (input.files.length) {
            handleFile(input.files[0], preview, area, callback);
        }
    });
}

function handleFile(file, previewEl, areaEl, callback) {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewEl.src = e.target.result;
            previewEl.style.display = 'block';
            areaEl.style.display = 'none';
            if (callback) callback();
        };
        reader.readAsDataURL(file);
    }
}

// Helper: Setup clear / reset action
function setupClear(clearBtnId, input, previewEl, uploadArea, resultEl, onClear) {
    const btn = document.getElementById(clearBtnId);
    if (!btn) return;
    btn.addEventListener('click', () => {
        if (input) input.value = '';
        if (previewEl) previewEl.style.display = 'none';
        if (uploadArea) uploadArea.style.display = 'block';
        if (resultEl) resultEl.style.display = 'none';
        if (onClear) onClear();
    });
}

// ----------------------------------------------------
// Attribute Analysis
// ----------------------------------------------------
const analyzeInput = document.getElementById('file-analyze');
const btnAnalyze = document.getElementById('btn-analyze');
const resAnalyze = document.getElementById('results-analyze');

setupUpload('upload-analyze', 'file-analyze', 'preview-analyze', () => {
    document.getElementById('preview-analyze-container').style.display = 'flex';
});

setupClear('clear-analyze', analyzeInput, document.getElementById('preview-analyze-container'), document.getElementById('upload-analyze'), resAnalyze);

btnAnalyze.addEventListener('click', async () => {
    if (!analyzeInput.files.length) return;
    const formData = new FormData();
    formData.append('file', analyzeInput.files[0]);

    UI.loading.style.display = 'flex';
    resAnalyze.style.display = 'none';

    try {
        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
            const face = Array.isArray(data.results) ? data.results[0] : data.results;
            const topGender = Object.keys(face.gender).reduce((a, b) => face.gender[a] > face.gender[b] ? a : b);
            resAnalyze.innerHTML = `
                <h3 style="margin-bottom:16px;font-family:'Syne',sans-serif;">Analysis Results</h3>
                <div class="result-card">
                    <div class="result-item"><div class="label">Gender</div><div class="value">${topGender}</div></div>
                    <div class="result-item"><div class="label">Emotion</div><div class="value" style="text-transform:capitalize">${face.dominant_emotion}</div></div>
                    <div class="result-item"><div class="label">Ethnicity</div><div class="value" style="text-transform:capitalize">${face.dominant_race}</div></div>
                </div>`;
            resAnalyze.style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (err) {
        alert('Failed to connect to server.');
    } finally {
        UI.loading.style.display = 'none';
    }
});

// ----------------------------------------------------
// Face Verification
// ----------------------------------------------------
const vInput1 = document.getElementById('file-verify-1');
const vInput2 = document.getElementById('file-verify-2');
const btnVerify = document.getElementById('btn-verify');
const resVerify = document.getElementById('results-verify');

const checkVerifyReady = () => {
    btnVerify.style.display = (vInput1.files.length && vInput2.files.length) ? 'inline-flex' : 'none';
};

setupUpload('upload-verify-1', 'file-verify-1', 'preview-verify-1', () => {
    document.getElementById('clear-verify-1').style.display = 'flex';
    checkVerifyReady();
});
setupUpload('upload-verify-2', 'file-verify-2', 'preview-verify-2', () => {
    document.getElementById('clear-verify-2').style.display = 'flex';
    checkVerifyReady();
});

setupClear('clear-verify-1', vInput1, document.getElementById('preview-verify-1'), document.getElementById('upload-verify-1'), resVerify, () => {
    document.getElementById('clear-verify-1').style.display = 'none';
    checkVerifyReady();
});
setupClear('clear-verify-2', vInput2, document.getElementById('preview-verify-2'), document.getElementById('upload-verify-2'), resVerify, () => {
    document.getElementById('clear-verify-2').style.display = 'none';
    checkVerifyReady();
});

btnVerify.addEventListener('click', async () => {
    const formData = new FormData();
    formData.append('img1', vInput1.files[0]);
    formData.append('img2', vInput2.files[0]);

    UI.loading.style.display = 'flex';
    resVerify.style.display = 'none';

    try {
        const res = await fetch('/api/verify', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
            const match = data.result.verified;
            const distance = data.result.distance.toFixed(4);
            const threshold = data.result.threshold;
            resVerify.innerHTML = `
                <div class="match-result ${match ? 'match-true' : 'match-false'}">
                    ${match ? '<i class="fa-solid fa-circle-check"></i> Same Person' : '<i class="fa-solid fa-circle-xmark"></i> Different People'}
                </div>
                <div style="margin-top:14px;text-align:center;color:var(--muted);font-size:0.88rem">
                    Distance: <strong style="color:var(--text)">${distance}</strong> &nbsp;|&nbsp; Threshold: <strong style="color:var(--text)">${threshold}</strong>
                </div>`;
            resVerify.style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (err) {
        alert('Failed to connect to server.');
    } finally {
        UI.loading.style.display = 'none';
    }
});

// ----------------------------------------------------
// Deepfake Detection
// ----------------------------------------------------
const dfInput = document.getElementById('file-deepfake');
const btnDeepfake = document.getElementById('btn-deepfake');
const resDeepfake = document.getElementById('results-deepfake');

setupUpload('upload-deepfake', 'file-deepfake', 'preview-deepfake', () => {
    document.getElementById('preview-deepfake-container').style.display = 'flex';
});

setupClear('clear-deepfake', dfInput, document.getElementById('preview-deepfake-container'), document.getElementById('upload-deepfake'), resDeepfake);

btnDeepfake.addEventListener('click', async () => {
    if (!dfInput.files.length) return;
    const formData = new FormData();
    formData.append('file', dfInput.files[0]);

    UI.loading.style.display = 'flex';
    resDeepfake.style.display = 'none';

    try {
        const res = await fetch('/api/deepfake', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
            const label = data.result.label.toLowerCase();
            const conf = (data.result.confidence * 100).toFixed(1);
            const isReal = label === 'real';
            resDeepfake.innerHTML = `
                <div class="match-result ${isReal ? 'match-true' : 'match-false'}\">
                    ${isReal
                        ? '<i class="fa-solid fa-shield-check"></i> Authentic Image'
                        : '<i class="fa-solid fa-triangle-exclamation"></i> Deepfake Detected'}
                </div>
                <div class="confidence-bar-wrap">
                    <div class="confidence-label">Confidence: <strong style="color:var(--text)">${conf}%</strong></div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${conf}%"></div>
                    </div>
                </div>`;
            resDeepfake.style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (err) {
        alert('Failed to connect to server.');
    } finally {
        UI.loading.style.display = 'none';
    }
});
