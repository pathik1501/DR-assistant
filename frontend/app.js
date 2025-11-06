// DR Assistant Frontend - JavaScript Application
// API URL: Use config.js (loaded before this script) or default to localhost for development
// config.js sets window.API_URL from Netlify environment variable
const API_URL = window.API_URL || 'http://localhost:8080';

// DOM Elements
const imageInput = document.getElementById('image-input');
const uploadArea = document.getElementById('upload-area');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const imageInfo = document.getElementById('image-info');
const includeExplanation = document.getElementById('include-explanation');
const includeHint = document.getElementById('include-hint');
const analyzeBtn = document.getElementById('analyze-btn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const apiStatus = document.getElementById('api-status');
const apiStatusIcon = document.getElementById('api-status-icon');
const apiStatusText = document.getElementById('api-status-text');

// State
let uploadedImage = null;
let analysisResult = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkApiConnection();
    setInterval(checkApiConnection, 10000); // Check every 10 seconds
    
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // File input
    imageInput.addEventListener('change', handleFileSelect);
    
    // Upload area drag and drop
    uploadArea.addEventListener('click', () => imageInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);
}

// API Connection Check
async function checkApiConnection() {
    try {
        const response = await fetch(`${API_URL}/health`, { 
            method: 'GET',
            timeout: 5000
        });
        
        if (response.ok) {
            apiStatus.className = 'api-status connected';
            apiStatusIcon.textContent = '🟢';
            apiStatusText.textContent = 'API Connected';
            analyzeBtn.disabled = !uploadedImage;
        } else {
            throw new Error('API not responding');
        }
    } catch (error) {
        apiStatus.className = 'api-status disconnected';
        apiStatusIcon.textContent = '🔴';
        apiStatusText.textContent = 'API Not Connected';
        analyzeBtn.disabled = true;
    }
}

// File Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    }
}

function processFile(file) {
    if (!file.type.match('image/(jpeg|jpg|png)')) {
        alert('Please upload a JPG, JPEG, or PNG image');
        return;
    }
    
    uploadedImage = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imagePreview.classList.remove('hidden');
        
        // Show image info
        imageInfo.innerHTML = `
            <strong>File:</strong> ${file.name}<br>
            <strong>Size:</strong> ${(file.size / 1024).toFixed(2)} KB<br>
            <strong>Type:</strong> ${file.type}
        `;
        
        // Enable analyze button if API is connected
        if (apiStatus.classList.contains('connected')) {
            analyzeBtn.disabled = false;
        }
    };
    reader.readAsDataURL(file);
}

// Image Analysis
async function analyzeImage() {
    if (!uploadedImage) {
        alert('Please upload an image first');
        return;
    }
    
    if (!apiStatus.classList.contains('connected')) {
        alert('API server is not connected. Please ensure the server is running on localhost:8080');
        return;
    }
    
    // Show loading
    loading.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    analyzeBtn.disabled = true;
    
    try {
        // Convert image to base64
        const base64Image = await fileToBase64(uploadedImage);
        
        // Prepare request
        const requestBody = {
            image_base64: base64Image.split(',')[1], // Remove data:image/... prefix
            include_explanation: includeExplanation.checked,
            include_hint: includeHint.checked
        };
        
        // Make API call with timeout
        const startTime = Date.now();
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 seconds
        
        let response;
        try {
            response = await fetch(`${API_URL}/predict_base64`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody),
                signal: controller.signal
            });
            clearTimeout(timeoutId);
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Request timeout: The analysis took too long (>120 seconds). Please try with a smaller image or check the API server.');
            }
            throw error;
        }
        
        if (!response.ok) {
            const errorText = await response.text();
            let errorData;
            try {
                errorData = JSON.parse(errorText);
            } catch {
                errorData = { detail: errorText };
            }
            throw new Error(errorData.detail || `API Error: ${response.status}`);
        }
        
        analysisResult = await response.json();
        const processingTime = (Date.now() - startTime) / 1000;
        
        // Display results
        displayResults(analysisResult);
        
    } catch (error) {
        console.error('Analysis error:', error);
        alert(`Error analyzing image: ${error.message}\n\nPlease ensure:\n1. API server is running\n2. Image format is correct\n3. Network connection is stable`);
    } finally {
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

// File to Base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Display Results
function displayResults(result) {
    // Display classification
    displayClassification(result);
    
    // Display heatmaps
    if (result.explanation && includeExplanation.checked) {
        displayHeatmaps(result.explanation);
    } else {
        document.getElementById('heatmaps-section').classList.add('hidden');
    }
    
    // Display recommendation
    if (result.clinical_hint && includeHint.checked) {
        displayRecommendation(result.clinical_hint);
    } else {
        document.getElementById('recommendation-section').classList.add('hidden');
    }
    
    // Show results section
    resultsSection.classList.remove('hidden');
    
    // Setup download
    setupDownload(result);
}

function displayClassification(result) {
    const card = document.getElementById('classification-card');
    const grade = result.prediction;
    
    // Set grade class
    card.className = `classification-card grade-${grade}`;
    
    // Grade icons
    const icons = {
        0: '✅',
        1: '⚠️',
        2: '🔶',
        3: '🔴',
        4: '🚨'
    };
    
    // Update content
    document.getElementById('grade-icon').textContent = icons[grade] || '✅';
    document.getElementById('grade-title').textContent = `Grade ${grade}`;
    document.getElementById('grade-description').textContent = result.grade_description;
    document.getElementById('confidence-value').textContent = `${(result.confidence * 100).toFixed(1)}%`;
    document.getElementById('processing-time').textContent = `Processing Time: ${result.processing_time.toFixed(2)}s`;
    
    // Update gauge
    updateConfidenceGauge(result.confidence);
}

function updateConfidenceGauge(confidence) {
    const percentage = confidence * 100;
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeText = document.getElementById('gauge-text');
    
    // Calculate angle for conic gradient
    const angle = (percentage / 100) * 360;
    
    gaugeFill.style.background = `conic-gradient(from 0deg, rgba(255,255,255,0.4) 0deg, rgba(255,255,255,0.8) ${angle}deg, transparent ${angle}deg, transparent 360deg)`;
    gaugeText.textContent = `${percentage.toFixed(0)}%`;
}

function displayHeatmaps(explanation) {
    const section = document.getElementById('heatmaps-section');
    const grid = document.getElementById('heatmaps-grid');
    
    // Clear previous
    grid.innerHTML = '';
    
    // Heatmap mappings
    const heatmaps = [
        { key: 'gradcam_heatmap_base64', title: 'Grad-CAM Heatmap' },
        { key: 'gradcam_overlay_base64', title: 'Grad-CAM Overlay' },
        { key: 'gradcam_plus_heatmap_base64', title: 'Grad-CAM++ Heatmap' },
        { key: 'gradcam_plus_overlay_base64', title: 'Grad-CAM++ Overlay' }
    ];
    
    let hasHeatmaps = false;
    
    heatmaps.forEach(heatmap => {
        if (explanation[heatmap.key]) {
            hasHeatmaps = true;
            const item = document.createElement('div');
            item.className = 'heatmap-item';
            item.innerHTML = `
                <h4>${heatmap.title}</h4>
                <img src="data:image/png;base64,${explanation[heatmap.key]}" alt="${heatmap.title}">
            `;
            grid.appendChild(item);
        }
    });
    
    if (hasHeatmaps) {
        section.classList.remove('hidden');
    } else {
        section.classList.add('hidden');
    }
}

function displayRecommendation(hint) {
    const section = document.getElementById('recommendation-section');
    const text = document.getElementById('recommendation-text');
    
    text.textContent = hint;
    section.classList.remove('hidden');
}

function setupDownload(result) {
    const downloadBtn = document.getElementById('download-btn');
    
    downloadBtn.onclick = () => {
        const report = {
            timestamp: new Date().toISOString(),
            prediction: result.prediction,
            grade: result.grade_description,
            confidence: result.confidence,
            clinical_hint: result.clinical_hint,
            processing_time: result.processing_time,
            explanation_available: result.explanation !== null
        };
        
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dr_analysis_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };
}

