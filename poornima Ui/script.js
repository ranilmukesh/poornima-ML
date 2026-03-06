/**
 * CardioSense+ Frontend Application
 * AI-Powered Stroke Risk Assessment Interface
 * 
 * Connects to FastAPI backend for predictions and SHAP explanations
 */

// API Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';

// DOM Elements
const elements = {
    loadingOverlay: document.getElementById('loadingOverlay'),
    patientForm: document.getElementById('patientForm'),
    submitBtn: document.getElementById('submitBtn'),
    resultsSection: document.getElementById('resultsSection'),
    assessmentForm: document.getElementById('assessmentForm'),
    backBtn: document.getElementById('backBtn'),
    riskCard: document.getElementById('riskCard'),
    riskPercentage: document.getElementById('riskPercentage'),
    riskLevel: document.getElementById('riskLevel'),
    riskConfidence: document.getElementById('riskConfidence'),
    progressRing: document.getElementById('progressRing'),
    meterPointer: document.getElementById('meterPointer'),
    factorsContainer: document.getElementById('factorsContainer'),
    recommendationsGrid: document.getElementById('recommendationsGrid'),
    whatifSection: document.getElementById('whatifSection'),
    whatifLoading: document.getElementById('whatifLoading'),
    whatifScenariosGrid: document.getElementById('whatifScenariosGrid'),
    whatifCombinedCard: document.getElementById('whatifCombinedCard'),
    combinedOriginalRisk: document.getElementById('combinedOriginalRisk'),
    combinedModifiedRisk: document.getElementById('combinedModifiedRisk'),
    combinedDelta: document.getElementById('combinedDelta')
};

// State
let currentPrediction = null;
let currentExplanation = null;
let currentWhatIf = null;
let currentFormData = null;

/**
 * Initialize the application
 */
function init() {
    setupEventListeners();
    checkAPIHealth();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Form submission
    elements.patientForm.addEventListener('submit', handleFormSubmit);

    // Back button
    elements.backBtn.addEventListener('click', showForm);

    // Input animations
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.closest('.input-group')?.classList.add('focused');
        });
        input.addEventListener('blur', () => {
            input.closest('.input-group')?.classList.remove('focused');
        });
    });
}

/**
 * Check API health status
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (!data.model_loaded) {
            showNotification('Model not loaded. Please run train_model.py first.', 'warning');
        }
    } catch (error) {
        showNotification('Cannot connect to API. Make sure the server is running.', 'error');
    }
}

/**
 * Handle form submission
 */
async function handleFormSubmit(e) {
    e.preventDefault();

    // Collect form data
    const formData = collectFormData();

    if (!validateFormData(formData)) {
        return;
    }

    // Show loading
    showLoading(true);

    try {
        // Make parallel API calls for prediction and explanation
        const [predictionResult, explanationResult] = await Promise.all([
            fetchPrediction(formData),
            fetchExplanation(formData)
        ]);

        currentPrediction = predictionResult;
        currentExplanation = explanationResult;
        currentFormData = formData;

        // Display results
        displayResults(predictionResult, explanationResult);

        // Fetch What-If analysis in background (non-blocking)
        fetchWhatIfAnalysis(formData);

    } catch (error) {
        console.error('API Error:', error);
        showNotification('Failed to get prediction. Please try again.', 'error');
        showLoading(false);
    }
}

/**
 * Collect form data
 */
function collectFormData() {
    return {
        gender: document.getElementById('gender').value,
        age: parseFloat(document.getElementById('age').value),
        hypertension: document.getElementById('hypertension').checked ? 1 : 0,
        heart_disease: document.getElementById('heart_disease').checked ? 1 : 0,
        ever_married: document.querySelector('input[name="ever_married"]:checked')?.value || '',
        work_type: document.getElementById('work_type').value,
        Residence_type: document.querySelector('input[name="Residence_type"]:checked')?.value || '',
        avg_glucose_level: parseFloat(document.getElementById('avg_glucose_level').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        smoking_status: document.getElementById('smoking_status').value
    };
}

/**
 * Validate form data
 */
function validateFormData(data) {
    const requiredFields = ['gender', 'age', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'];

    for (const field of requiredFields) {
        if (!data[field] && data[field] !== 0) {
            showNotification(`Please fill in all required fields.`, 'warning');
            return false;
        }
    }

    if (data.age < 0 || data.age > 120) {
        showNotification('Please enter a valid age (0-120).', 'warning');
        return false;
    }

    if (data.bmi < 0 || data.bmi > 100) {
        showNotification('Please enter a valid BMI (0-100).', 'warning');
        return false;
    }

    return true;
}

/**
 * Fetch prediction from API
 */
async function fetchPrediction(data) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
}

/**
 * Fetch explanation from API
 */
async function fetchExplanation(data) {
    const response = await fetch(`${API_BASE_URL}/explain`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
}

/**
 * Display results
 */
function displayResults(prediction, explanation) {
    // Hide loading after a short delay for smooth transition
    setTimeout(() => {
        showLoading(false);

        // Switch views
        elements.assessmentForm.classList.add('hidden');
        elements.resultsSection.classList.remove('hidden');

        // Scroll to top smoothly
        window.scrollTo({ top: 0, behavior: 'smooth' });

        // Animate results
        setTimeout(() => {
            animateRiskScore(prediction);
            displayFactors(explanation.top_contributing_factors);
            displayRecommendations(prediction.risk_level);

            // Initialize Risk Simulator
            initSimulator();

            // Show chat widget after results are displayed
            showChatWidget();
        }, 300);

    }, 1000);
}

/**
 * Animate risk score display
 */
function animateRiskScore(prediction) {
    const percentage = prediction.probability_percentage;
    const riskLevel = prediction.risk_level;

    // Set risk card class
    elements.riskCard.className = 'risk-card ' + riskLevel.toLowerCase();

    // Update confidence
    elements.riskConfidence.textContent = prediction.confidence;

    // Animate percentage counter
    animateCounter(elements.riskPercentage, 0, percentage, 1500);

    // Animate progress ring
    const circumference = 2 * Math.PI * 54; // r = 54
    const offset = circumference - (percentage / 100) * circumference;

    setTimeout(() => {
        elements.progressRing.style.strokeDashoffset = offset;
    }, 100);

    // Animate risk level text
    setTimeout(() => {
        elements.riskLevel.textContent = riskLevel;
    }, 500);

    // Animate meter pointer
    setTimeout(() => {
        elements.meterPointer.style.left = `${percentage}%`;
    }, 100);
}

/**
 * Animate counter from start to end
 */
function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    const diff = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out)
        const easeOut = 1 - Math.pow(1 - progress, 3);

        const current = start + diff * easeOut;
        element.textContent = current.toFixed(1);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/**
 * Display factor cards
 */
function displayFactors(factors) {
    elements.factorsContainer.innerHTML = '';

    // Find max impact for normalization
    const maxImpact = Math.max(...factors.map(f => Math.abs(f.impact)));

    factors.forEach((factor, index) => {
        const card = createFactorCard(factor, maxImpact);
        elements.factorsContainer.appendChild(card);

        // Trigger animation
        setTimeout(() => {
            card.classList.add('animate');
        }, 50);
    });
}

/**
 * Create a factor card element
 */
function createFactorCard(factor, maxImpact) {
    const card = document.createElement('div');
    card.className = 'factor-card';

    const isPositive = factor.impact > 0;
    const normalizedImpact = (Math.abs(factor.impact) / maxImpact) * 100;
    const featureName = formatFeatureName(factor.feature);

    card.innerHTML = `
        <div class="factor-header">
            <span class="factor-name">${featureName}</span>
            <span class="factor-direction ${isPositive ? 'increases' : 'reduces'}">
                ${factor.direction}
            </span>
        </div>
        <p class="factor-interpretation">${factor.interpretation}</p>
        <div class="factor-bar">
            <div class="factor-bar-fill ${isPositive ? 'positive' : 'negative'}" 
                 style="width: 0%"
                 data-width="${normalizedImpact}%"></div>
        </div>
    `;

    // Animate bar after card is added
    setTimeout(() => {
        const bar = card.querySelector('.factor-bar-fill');
        bar.style.width = `${normalizedImpact}%`;
    }, 300);

    return card;
}

/**
 * Format feature name for display
 */
function formatFeatureName(name) {
    // Handle common feature name patterns
    const nameMap = {
        'age': 'Age',
        'bmi': 'BMI (Body Mass Index)',
        'avg_glucose_level': 'Average Glucose Level',
        'hypertension_0': 'No Hypertension',
        'hypertension_1': 'Has Hypertension',
        'heart_disease_0': 'No Heart Disease',
        'heart_disease_1': 'Has Heart Disease',
        'gender_Male': 'Gender: Male',
        'gender_Female': 'Gender: Female',
        'ever_married_Yes': 'Married',
        'ever_married_No': 'Not Married',
        'work_type_Private': 'Private Sector Work',
        'work_type_Self-employed': 'Self-Employed',
        'work_type_Govt_job': 'Government Job',
        'work_type_children': 'Children/Student',
        'work_type_Never_worked': 'Never Worked',
        'Residence_type_Urban': 'Urban Residence',
        'Residence_type_Rural': 'Rural Residence',
        'smoking_status_formerly smoked': 'Former Smoker',
        'smoking_status_never smoked': 'Never Smoked',
        'smoking_status_smokes': 'Current Smoker',
        'smoking_status_Unknown': 'Smoking: Unknown'
    };

    return nameMap[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Display recommendations based on risk level
 */
function displayRecommendations(riskLevel) {
    const recommendations = getRecommendations(riskLevel);
    elements.recommendationsGrid.innerHTML = '';

    recommendations.forEach(rec => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';

        card.innerHTML = `
            <div class="recommendation-icon">${rec.icon}</div>
            <h4 class="recommendation-title">${rec.title}</h4>
            <p class="recommendation-text">${rec.text}</p>
        `;

        elements.recommendationsGrid.appendChild(card);
    });
}

/**
 * Get recommendations based on risk level
 */
function getRecommendations(riskLevel) {
    const baseRecommendations = [
        {
            icon: '🏃',
            title: 'Regular Exercise',
            text: 'Aim for 150 minutes of moderate aerobic activity weekly to maintain cardiovascular health.'
        },
        {
            icon: '🥗',
            title: 'Balanced Diet',
            text: 'Focus on fruits, vegetables, whole grains, and limit sodium and saturated fats.'
        }
    ];

    if (riskLevel === 'HIGH') {
        return [
            {
                icon: '🏥',
                title: 'Consult a Doctor',
                text: 'Schedule an appointment with a healthcare provider for comprehensive evaluation.'
            },
            {
                icon: '📊',
                title: 'Monitor Vitals',
                text: 'Regularly track blood pressure, glucose levels, and maintain health records.'
            },
            ...baseRecommendations
        ];
    } else if (riskLevel === 'MEDIUM') {
        return [
            {
                icon: '📋',
                title: 'Regular Checkups',
                text: 'Schedule regular health screenings to monitor your risk factors.'
            },
            {
                icon: '🚭',
                title: 'Lifestyle Changes',
                text: 'Consider quitting smoking and reducing alcohol consumption if applicable.'
            },
            ...baseRecommendations
        ];
    } else {
        return [
            {
                icon: '✅',
                title: 'Keep It Up!',
                text: 'Continue maintaining your healthy lifestyle and regular health practices.'
            },
            {
                icon: '🧘',
                title: 'Stress Management',
                text: 'Practice mindfulness and stress-reduction techniques for overall wellness.'
            },
            ...baseRecommendations
        ];
    }
}

/**
 * Show/hide loading overlay
 */
function showLoading(show) {
    if (show) {
        elements.loadingOverlay.classList.add('active');
    } else {
        elements.loadingOverlay.classList.remove('active');
    }
}

/**
 * Show form and hide results
 */
function showForm() {
    elements.resultsSection.classList.add('hidden');
    elements.assessmentForm.classList.remove('hidden');

    // Reset form
    elements.patientForm.reset();

    // Reset What-If state
    resetWhatIf();

    // Hide chat widget
    hideChatWidget();

    // Scroll to form
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Show notification toast
 */
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelector('.notification');
    if (existing) {
        existing.remove();
    }

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span class="notification-message">${message}</span>
        <button class="notification-close">&times;</button>
    `;

    // Add styles dynamically
    notification.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: ${type === 'error' ? '#E85D4C' : type === 'warning' ? '#FF9800' : '#2D9596'};
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 10000;
        max-width: 90%;
        animation: slideUp 0.3s ease;
    `;

    // Add animation keyframes
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateX(-50%) translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }
        }
    `;
    document.head.appendChild(style);

    document.body.appendChild(notification);

    // Close button handler
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.remove();
    });

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(-50%) translateY(20px)';
        notification.style.transition = 'all 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

/**
 * Demo mode - fill form with sample data
 */
function fillDemoData() {
    document.getElementById('gender').value = 'Male';
    document.getElementById('age').value = '67';
    document.getElementById('hypertension').checked = false;
    document.getElementById('heart_disease').checked = true;
    document.getElementById('married_yes').checked = true;
    document.getElementById('urban').checked = true;
    document.getElementById('avg_glucose_level').value = '228.69';
    document.getElementById('bmi').value = '36.6';
    document.getElementById('work_type').value = 'Private';
    document.getElementById('smoking_status').value = 'formerly smoked';
}

/**
 * =============================================
 * WHAT-IF SCENARIO ANALYSIS
 * =============================================
 */

/**
 * Fetch What-If analysis from API
 */
async function fetchWhatIfAnalysis(formData) {
    // Show the What-If section with loading state
    elements.whatifSection.style.display = 'block';
    elements.whatifLoading.style.display = 'flex';
    elements.whatifScenariosGrid.innerHTML = '';
    elements.whatifCombinedCard.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}/whatif`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const whatifData = await response.json();
        currentWhatIf = whatifData;

        // Hide loading, display results
        elements.whatifLoading.style.display = 'none';
        displayWhatIfAnalysis(whatifData);

    } catch (error) {
        console.error('What-If API Error:', error);
        elements.whatifLoading.style.display = 'none';
        elements.whatifScenariosGrid.innerHTML = `
            <div class="whatif-error">
                <p>⚠️ Could not generate What-If scenarios. The analysis will still work without this feature.</p>
            </div>
        `;
    }
}

/**
 * Display What-If analysis results
 */
function displayWhatIfAnalysis(data) {
    if (!data.scenarios || data.scenarios.length === 0) {
        elements.whatifScenariosGrid.innerHTML = `
            <div class="whatif-empty">
                <p>✅ Your current health parameters are already in healthy ranges. No significant changes to suggest!</p>
            </div>
        `;
        return;
    }

    // Render scenario cards with staggered animation
    data.scenarios.forEach((scenario, index) => {
        const card = createScenarioCard(scenario);
        elements.whatifScenariosGrid.appendChild(card);

        // Staggered fade-in animation
        setTimeout(() => {
            card.classList.add('animate');
        }, 100 + index * 150);
    });

    // Show combined outcome if available
    if (data.combined_risk !== null && data.combined_risk !== undefined && data.scenarios.length > 1) {
        setTimeout(() => {
            displayCombinedOutcome(data);
        }, 100 + data.scenarios.length * 150 + 200);
    }
}

/**
 * Create a single scenario card
 */
function createScenarioCard(scenario) {
    const card = document.createElement('div');
    card.className = 'whatif-card';

    const isReduction = scenario.risk_delta > 0;
    const deltaAbs = Math.abs(scenario.risk_delta).toFixed(1);
    const reductionPct = Math.abs(scenario.risk_reduction_percent).toFixed(1);

    // Determine the bar width (scale delta to percentage of original risk for visualization)
    const barWidth = Math.min(Math.abs(scenario.risk_reduction_percent), 100);

    card.innerHTML = `
        <div class="whatif-card-icon">${scenario.icon}</div>
        <div class="whatif-card-content">
            <h4 class="whatif-card-title">${scenario.title}</h4>
            <p class="whatif-card-desc">${scenario.description}</p>
            
            <div class="whatif-card-comparison">
                <div class="whatif-risk-original">
                    <span class="whatif-risk-label">Current</span>
                    <span class="whatif-risk-val">${scenario.original_risk.toFixed(1)}%</span>
                </div>
                <div class="whatif-arrow-container">
                    <svg class="whatif-arrow ${isReduction ? 'reduction' : 'increase'}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                </div>
                <div class="whatif-risk-modified">
                    <span class="whatif-risk-label">Modified</span>
                    <span class="whatif-risk-val ${isReduction ? 'improved' : 'worsened'}">${scenario.modified_risk.toFixed(1)}%</span>
                </div>
            </div>
            
            <div class="whatif-delta-bar">
                <div class="whatif-delta-fill ${isReduction ? 'positive' : 'negative'}" 
                     style="width: 0%" data-width="${barWidth}%"></div>
            </div>
            
            <div class="whatif-delta-text ${isReduction ? 'positive' : 'negative'}">
                ${isReduction ? '↓' : '↑'} ${deltaAbs}% risk ${isReduction ? 'reduction' : 'increase'}
                <span class="whatif-delta-pct">(${reductionPct}% ${isReduction ? 'improvement' : 'change'})</span>
            </div>
        </div>
    `;

    // Animate the delta bar after card is rendered
    setTimeout(() => {
        const bar = card.querySelector('.whatif-delta-fill');
        if (bar) {
            bar.style.width = `${barWidth}%`;
        }
    }, 500);

    return card;
}

/**
 * Display combined best-case outcome card
 */
function displayCombinedOutcome(data) {
    elements.whatifCombinedCard.style.display = 'block';

    const originalRisk = data.original_risk;
    const combinedRisk = data.combined_risk;
    const totalDelta = originalRisk - combinedRisk;
    const totalReductionPct = originalRisk > 0 ? (totalDelta / originalRisk * 100) : 0;

    elements.combinedOriginalRisk.textContent = `${originalRisk.toFixed(1)}%`;
    elements.combinedModifiedRisk.textContent = `${combinedRisk.toFixed(1)}%`;

    // Color the modified risk value
    if (combinedRisk < originalRisk) {
        elements.combinedModifiedRisk.classList.add('improved');
    }

    // Delta summary
    const isReduction = totalDelta > 0;
    elements.combinedDelta.innerHTML = `
        <div class="combined-delta-badge ${isReduction ? 'positive' : 'negative'}">
            ${isReduction ? '↓' : '↑'} ${Math.abs(totalDelta).toFixed(1)}% total risk ${isReduction ? 'reduction' : 'increase'}
            <span>(${Math.abs(totalReductionPct).toFixed(1)}% overall ${isReduction ? 'improvement' : 'change'})</span>
        </div>
        <p class="combined-risk-level">Risk Level: <strong>${data.original_risk_level}</strong> → <strong class="${(data.combined_risk_level || '').toLowerCase()}">${data.combined_risk_level || 'N/A'}</strong></p>
    `;

    // Animate in
    setTimeout(() => {
        elements.whatifCombinedCard.classList.add('animate');
    }, 100);
}

/**
 * Reset What-If section state
 */
function resetWhatIf() {
    currentWhatIf = null;
    elements.whatifSection.style.display = 'none';
    elements.whatifScenariosGrid.innerHTML = '';
    elements.whatifCombinedCard.style.display = 'none';
    elements.whatifCombinedCard.classList.remove('animate');
    elements.combinedModifiedRisk.classList.remove('improved');
}

/**
 * =============================================
 * INTERACTIVE RISK SIMULATOR
 * =============================================
 */

let simTimeout = null;

function initSimulator() {
    if (!currentFormData || !currentPrediction) return;

    // Show the simulator section
    const simCard = document.getElementById('interactiveSimulator');
    if (simCard) simCard.classList.remove('hidden');

    // Populate baseline stats
    const baselineRiskStr = currentPrediction.probability_percentage.toFixed(1) + '%';
    document.getElementById('simBaselineStat').textContent = baselineRiskStr;

    // Initial UI update with baseline data
    updateSimulatorUI(currentPrediction, currentPrediction, currentFormData);

    // Setup Event Listeners
    setupSimControl('Age', 'age', 20, 100);
    setupSimControl('Bmi', 'bmi', 15, 60);
    setupSimControl('Glucose', 'avg_glucose_level', 50, 300);

    // Toggle Setup
    const toggle = document.getElementById('simSmokerToggle');
    toggle.checked = (currentFormData.smoking_status === 'smokes' || currentFormData.smoking_status === 'formerly smoked');
    toggle.onchange = () => { triggerSimulate(); };
}

function setupSimControl(idPrefix, fieldName, min, max) {
    const slider = document.getElementById(`sim${idPrefix}Slider`);
    const num = document.getElementById(`sim${idPrefix}Num`);

    if (slider && num && currentFormData) {
        let val = Number(currentFormData[fieldName]) || min;
        if (val < min) val = min;
        if (val > max) val = max;

        slider.value = val;
        num.value = val;

        slider.oninput = (e) => { num.value = e.target.value; triggerSimulate(); };
        num.oninput = (e) => {
            let v = Number(e.target.value);
            if (v >= min && v <= max) { slider.value = v; triggerSimulate(); }
        };
    }
}

function triggerSimulate() {
    clearTimeout(simTimeout);
    simTimeout = setTimeout(simulateRisk, 300); // 300ms debounce
}

async function simulateRisk() {
    if (!currentFormData) return;

    // Create new patient data object
    const simulatedData = { ...currentFormData };

    // Override with simulator values
    simulatedData.age = parseFloat(document.getElementById('simAgeNum').value);
    simulatedData.bmi = parseFloat(document.getElementById('simBmiNum').value);
    simulatedData.avg_glucose_level = parseFloat(document.getElementById('simGlucoseNum').value);

    const isSmoker = document.getElementById('simSmokerToggle').checked;

    // Map simplified toggle back to model categories
    if (isSmoker) {
        simulatedData.smoking_status = 'smokes';
    } else {
        simulatedData.smoking_status = 'never smoked';
    }

    try {
        const response = await fetchPrediction(simulatedData);
        updateSimulatorUI(currentPrediction, response, simulatedData);
    } catch (e) {
        console.error("Simulation failed", e);
    }
}

function updateSimulatorUI(baselinePred, targetPred, simData) {
    const risk = targetPred.probability_percentage;
    const levelStr = targetPred.risk_level.toUpperCase();

    // Text values
    document.getElementById('simRiskValue').textContent = risk.toFixed(1) + '%';
    document.getElementById('simRiskLabel').textContent = levelStr + ' RISK';
    document.getElementById('simTargetStat').textContent = risk.toFixed(1) + '%';

    // Delta
    const delta = (risk - baselinePred.probability_percentage).toFixed(1);
    const deltaEl = document.getElementById('simDeltaStat');
    deltaEl.textContent = (delta > 0 ? '+' : '') + delta + '%';
    deltaEl.className = 'sim-stat-val ' + (delta > 0 ? 'negative' : (delta < 0 ? 'positive' : ''));

    // Dynamic Arc length = 219.91
    const trackFilled = 219.91 * (1 - (risk / 100)); // risk mapped as 0-100%
    const simTrack = document.getElementById('simTrack');
    if (simTrack) {
        simTrack.style.strokeDashoffset = trackFilled;

        // Color
        let color = 'var(--risk-low)';
        if (targetPred.risk_level === 'MEDIUM') color = 'var(--risk-medium)';
        if (targetPred.risk_level === 'HIGH') color = 'var(--risk-high)';
        simTrack.style.stroke = color;
    }

    // Baseline Marker (Angle 0 is left, angle 180 is right)
    const baseRisk = baselinePred.probability_percentage;
    const baseAngle = 180 * (baseRisk / 100);
    const baseGroup = document.getElementById('simBaselineGroup');
    const baseTextGroup = document.getElementById('simBaselineTextGroup');
    if (baseGroup) baseGroup.style.transform = `rotate(${baseAngle}deg)`;
    if (baseTextGroup) baseTextGroup.style.transform = `translate(18px, 90px) rotate(${-baseAngle}deg)`;

    // Target Marker
    const targetAngle = 180 * (risk / 100);
    const targetGroup = document.getElementById('simTargetGroup');
    const targetTextGroup = document.getElementById('simTargetTextGroup');
    if (targetGroup) targetGroup.style.transform = `rotate(${targetAngle}deg)`;
    if (targetTextGroup) targetTextGroup.style.transform = `translate(18px, 110px) rotate(${-targetAngle}deg)`;
}


/**
 * =============================================
 * AI CHAT FUNCTIONALITY
 * =============================================
 */

let chatSessionId = null;
let chatIsOpen = false;
let chatInitialized = false;
let chatElems = {};

function initChatElements() {
    chatElems = {
        widget: document.getElementById('chatWidget'),
        toggle: document.getElementById('chatToggle'),
        panel: document.getElementById('chatPanel'),
        minimize: document.getElementById('chatMinimize'),
        messages: document.getElementById('chatMessages'),
        input: document.getElementById('chatInput'),
        send: document.getElementById('chatSend'),
        status: document.getElementById('chatStatus'),
        iconOpen: document.querySelector('.chat-icon-open'),
        iconClose: document.querySelector('.chat-icon-close'),
    };

    chatElems.toggle.addEventListener('click', toggleChat);
    chatElems.minimize.addEventListener('click', toggleChat);
    chatElems.send.addEventListener('click', sendChatMessage);
    chatElems.input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
    chatElems.input.addEventListener('input', () => {
        chatElems.send.disabled = !chatElems.input.value.trim();
    });
}

function showChatWidget() {
    if (!chatElems.widget) initChatElements();
    chatElems.widget.style.display = 'block';
}

function hideChatWidget() {
    if (!chatElems.widget) return;
    chatElems.widget.style.display = 'none';
    chatElems.panel.style.display = 'none';
    chatIsOpen = false;
    chatInitialized = false;
    chatSessionId = null;
    if (chatElems.messages) chatElems.messages.innerHTML = '';
    if (chatElems.iconOpen) chatElems.iconOpen.style.display = 'block';
    if (chatElems.iconClose) chatElems.iconClose.style.display = 'none';
}

function toggleChat() {
    chatIsOpen = !chatIsOpen;
    chatElems.panel.style.display = chatIsOpen ? 'flex' : 'none';
    chatElems.iconOpen.style.display = chatIsOpen ? 'none' : 'block';
    chatElems.iconClose.style.display = chatIsOpen ? 'block' : 'none';

    if (chatIsOpen && !chatInitialized) {
        initializeChat();
    }
    if (chatIsOpen) {
        chatElems.input.focus();
    }
}

async function initializeChat() {
    chatInitialized = true;
    addTypingIndicator();
    setChatStatus('Connecting...');

    const formData = collectFormData();

    try {
        const response = await fetch(`${API_BASE_URL}/chat/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_data: formData,
                prediction: currentPrediction,
                explanation: currentExplanation,
                whatif: currentWhatIf || {}
            })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        chatSessionId = data.session_id;

        removeTypingIndicator();
        addChatMessage('ai', data.message);
        setChatStatus('Online');

    } catch (error) {
        console.error('Chat init error:', error);
        removeTypingIndicator();
        addChatMessage('system',
            '⚠️ Could not connect to CardioSense AI. Make sure NVIDIA_API_KEY is set and agno is installed.'
        );
        setChatStatus('Offline');
    }
}

async function sendChatMessage() {
    const message = chatElems.input.value.trim();
    if (!message || !chatSessionId) return;

    addChatMessage('user', message);
    chatElems.input.value = '';
    chatElems.send.disabled = true;

    addTypingIndicator();
    setChatStatus('Thinking...');

    try {
        const response = await fetch(`${API_BASE_URL}/chat/message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: chatSessionId,
                message: message
            })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        removeTypingIndicator();
        addChatMessage('ai', data.response);
        setChatStatus('Online');

    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator();
        addChatMessage('system', '⚠️ Failed to get a response. Please try again.');
        setChatStatus('Online');
    }
}

function addChatMessage(type, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-msg chat-msg-${type}`;

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    bubble.innerHTML = formatChatMarkdown(text);

    msgDiv.appendChild(bubble);
    chatElems.messages.appendChild(msgDiv);
    chatElems.messages.scrollTop = chatElems.messages.scrollHeight;
}

function addTypingIndicator() {
    if (document.getElementById('chatTyping')) return;
    const indicator = document.createElement('div');
    indicator.id = 'chatTyping';
    indicator.className = 'chat-msg chat-msg-ai';
    indicator.innerHTML = `
        <div class="chat-bubble typing-indicator">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
    `;
    chatElems.messages.appendChild(indicator);
    chatElems.messages.scrollTop = chatElems.messages.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById('chatTyping');
    if (el) el.remove();
}

function setChatStatus(text) {
    if (chatElems.status) {
        chatElems.status.innerHTML = `<span class="status-dot"></span> ${text}`;
    }
}

function formatChatMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/^\s*[-•]\s+(.+)/gm, '<li>$1</li>')
        .replace(/^\s*(\d+)\.\s+(.+)/gm, '<li>$2</li>')
        .replace(/\n/g, '<br>');
}


// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', init);

// Expose demo function globally for testing
window.fillDemoData = fillDemoData;
