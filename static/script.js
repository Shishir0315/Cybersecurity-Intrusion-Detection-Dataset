document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultView = document.getElementById('result-view');
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numerical values
    const numericFields = ['network_packet_size', 'login_attempts', 'session_duration', 'ip_reputation_score', 'failed_logins', 'unusual_time_access'];
    numericFields.forEach(field => {
        data[field] = parseFloat(data[field]);
    });

    // Loading state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = 'ANALYZING...';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.error) {
            alert('Error: ' + result.error);
            return;
        }

        // Update UI
        displayResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to server');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = 'RECONSTRUCT & ANALYZE';
    }
});

function displayResult(result) {
    const resultView = document.getElementById('result-view');
    const statusBox = document.getElementById('status-box');
    const statusText = document.getElementById('status-text');
    const statusIcon = document.getElementById('status-icon');
    const mseVal = document.getElementById('mse-val');
    const confVal = document.getElementById('conf-val');
    const meterFill = document.getElementById('meter-fill');
    
    resultView.classList.remove('hidden');
    
    mseVal.innerText = result.mse;
    confVal.innerText = result.confidence + '%';
    
    // Calculate meter width (up to a limit)
    let meterWidth = (result.mse / 0.1) * 100; // 0.1 is 2x threshold
    meterWidth = Math.min(meterWidth, 100);
    meterFill.style.width = meterWidth + '%';
    
    if (result.is_anomaly) {
        statusBox.className = 'result-status status-anomaly';
        statusText.innerText = 'ANOMALY DETECTED';
        statusIcon.innerText = '⚠️';
        meterFill.style.backgroundColor = '#ff3366';
    } else {
        statusBox.className = 'result-status status-normal';
        statusText.innerText = 'NORMAL TRAFFIC';
        statusIcon.innerText = '✅';
        meterFill.style.backgroundColor = '#00f2fe';
    }
}
