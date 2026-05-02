document.addEventListener('DOMContentLoaded', () => {
    // === Tab Switching ===
    const tabSingle = document.getElementById('tab-single');
    const tabBatch = document.getElementById('tab-batch');
    const sectionSingle = document.getElementById('section-single');
    const sectionBatch = document.getElementById('section-batch');

    tabSingle.addEventListener('click', () => {
        tabSingle.className = "flex-1 py-4 px-6 text-center font-medium tab-active transition-colors";
        tabBatch.className = "flex-1 py-4 px-6 text-center font-medium tab-inactive transition-colors";
        sectionSingle.classList.remove('hidden');
        sectionBatch.classList.add('hidden');
    });

    tabBatch.addEventListener('click', () => {
        tabBatch.className = "flex-1 py-4 px-6 text-center font-medium tab-active transition-colors";
        tabSingle.className = "flex-1 py-4 px-6 text-center font-medium tab-inactive transition-colors";
        sectionBatch.classList.remove('hidden');
        sectionSingle.classList.add('hidden');
    });

    // === Single Predict Logic ===
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const filenameDisplay = document.getElementById('filename');
    const removeBtn = document.getElementById('remove-btn');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    const resultArea = document.getElementById('result-area');
    const successResult = document.getElementById('success-result');
    const errorResult = document.getElementById('error-result');
    const energyValue = document.getElementById('energy-value');
    const errorMessage = document.getElementById('error-message');

    let currentFile = null;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-active'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-active'), false);
    });

    dropZone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files), false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function() { handleFiles(this.files); });

    function handleFiles(files) {
        if (files.length > 0) {
            currentFile = files[0];
            filenameDisplay.textContent = currentFile.name;
            dropZone.classList.add('hidden');
            fileInfo.classList.remove('hidden');
            predictBtn.disabled = false;
            resultArea.classList.add('hidden');
        }
    }

    removeBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        predictBtn.disabled = true;
        resultArea.classList.add('hidden');
    });

    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        predictBtn.disabled = true;
        btnText.textContent = '预测中...';
        btnSpinner.classList.remove('hidden');
        resultArea.classList.add('hidden');
        successResult.classList.add('hidden');
        errorResult.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            resultArea.classList.remove('hidden');

            if (data.status === 'success') {
                energyValue.textContent = data.formation_energy.toFixed(4);
                successResult.classList.remove('hidden');
            } else {
                errorMessage.textContent = data.message || '服务器返回错误状态';
                errorResult.classList.remove('hidden');
            }
        } catch (error) {
            resultArea.classList.remove('hidden');
            errorMessage.textContent = '网络请求失败，请确保后端服务正在运行。';
            errorResult.classList.remove('hidden');
        } finally {
            predictBtn.disabled = false;
            btnText.textContent = '开始预测';
            btnSpinner.classList.add('hidden');
        }
    });

    // === Batch Predict Logic ===
    const batchPathInput = document.getElementById('batch-path-input');
    const batchFileInput = document.getElementById('batch-file-input');
    const batchPredictBtn = document.getElementById('batch-predict-btn');
    const batchBtnText = document.getElementById('batch-btn-text');
    const batchBtnSpinner = document.getElementById('batch-btn-spinner');
    const batchProgressArea = document.getElementById('batch-progress-area');
    const batchProgressText = document.getElementById('batch-progress-text');
    const batchProgressPct = document.getElementById('batch-progress-pct');
    const batchProgressBar = document.getElementById('batch-progress-bar');
    const batchResultArea = document.getElementById('batch-result-area');
    const batchTbody = document.getElementById('batch-tbody');
    const batchErrorArea = document.getElementById('batch-error-area');
    const batchDownloadBtn = document.getElementById('batch-download-btn');

    let batchResults = [];

    batchPredictBtn.addEventListener('click', async () => {
        const pathVal = batchPathInput.value.trim();
        const files = batchFileInput.files;

        if (!pathVal && files.length === 0) {
            alert('请输入文件路径或上传文件！');
            return;
        }

        // Reset UI
        batchPredictBtn.disabled = true;
        batchBtnText.textContent = '处理中...';
        batchBtnSpinner.classList.remove('hidden');
        batchProgressArea.classList.remove('hidden');
        batchResultArea.classList.remove('hidden');
        batchErrorArea.classList.add('hidden');
        batchDownloadBtn.classList.add('hidden');
        batchTbody.innerHTML = '';
        batchProgressBar.style.width = '0%';
        batchProgressPct.textContent = '0%';
        batchProgressText.textContent = '准备中...';
        batchResults = [];

        const formData = new FormData();
        if (pathVal) {
            formData.append('file_path', pathVal);
        } else if (files.length > 0) {
            formData.append('file', files[0]);
        }

        try {
            const response = await fetch('/predict_batch_stream', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                
                // Keep the last incomplete line in buffer
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.status === 'start') {
                            batchProgressText.textContent = `开始预测，共 ${data.total} 条数据`;
                        } else if (data.status === 'progress') {
                            const pct = Math.round((data.current / data.total) * 100);
                            batchProgressBar.style.width = `${pct}%`;
                            batchProgressPct.textContent = `${pct}%`;
                            batchProgressText.textContent = `预测中... (${data.current}/${data.total})`;
                            
                            // Add to table (limit to showing last 100 to prevent DOM lag)
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${data.result.id}</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${data.result.formula}</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${data.result.formation_energy.toFixed(4)}</td>
                            `;
                            batchTbody.prepend(tr);
                            if (batchTbody.children.length > 100) {
                                batchTbody.removeChild(batchTbody.lastChild);
                            }
                            
                            batchResults.push(data.result);
                        } else if (data.status === 'progress_error') {
                            console.error(`Item ${data.current} error: ${data.message}`);
                        } else if (data.status === 'done') {
                            batchProgressText.textContent = '预测完成！';
                            batchProgressBar.style.width = '100%';
                            batchProgressPct.textContent = '100%';
                            batchDownloadBtn.classList.remove('hidden');
                        } else if (data.status === 'error') {
                            throw new Error(data.message);
                        }
                    } catch (e) {
                        console.error('JSON parse error:', e, line);
                    }
                }
            }
        } catch (error) {
            batchErrorArea.textContent = '发生错误：' + error.message;
            batchErrorArea.classList.remove('hidden');
        } finally {
            batchPredictBtn.disabled = false;
            batchBtnText.textContent = '开始批量预测';
            batchBtnSpinner.classList.add('hidden');
        }
    });

    // CSV Download
    batchDownloadBtn.addEventListener('click', () => {
        if (batchResults.length === 0) return;
        
        const headers = ['ID', 'Formula', 'Formation_Energy'];
        const csvRows = [headers.join(',')];
        
        for (const res of batchResults) {
            csvRows.push(`${res.id},${res.formula},${res.formation_energy}`);
        }
        
        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', 'batch_predictions.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
