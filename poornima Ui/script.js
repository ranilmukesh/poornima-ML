/**
 * DiabeSense+ Frontend Application
 * AI-Powered Diabetes HbA1c Prediction Interface
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

function init() {
    setupEventListeners();
    checkAPIHealth();
}

function setupEventListeners() {
    elements.patientForm.addEventListener('submit', handleFormSubmit);
    elements.backBtn.addEventListener('click', showForm);

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

async function handleFormSubmit(e) {
    e.preventDefault();
    const formData = collectFormData();
    if (!validateFormData(formData)) return;

    showLoading(true);

    try {
        const [predictionResult, explanationResult] = await Promise.all([
            fetchPrediction(formData),
            fetchExplanation(formData)
        ]);

        currentPrediction = predictionResult;
        currentExplanation = explanationResult;
        currentFormData = formData;

        displayResults(predictionResult, explanationResult);
        fetchWhatIfAnalysis(formData);
    } catch (error) {
        console.error('API Error:', error);
        showNotification('Failed to get prediction. Please try again.', 'error');
        showLoading(false);
    }
}

/**
 * Collect all 36 diabetes form fields
 */
function collectFormData() {
    return {
        PostBLAge: parseFloat(document.getElementById('PostBLAge').value),
        PreBLGender: document.querySelector('input[name="PreBLGender"]:checked')?.value || '',
        PreRarea: parseInt(document.querySelector('input[name="PreRarea"]:checked')?.value || '0'),
        PreRmaritalstatus: parseFloat(document.getElementById('PreRmaritalstatus').value),
        PreReducation: parseFloat(document.getElementById('PreReducation').value),
        PreRpresentoccupation: parseFloat(document.getElementById('PreRpresentoccupation').value),
        PreRdiafather: parseInt(document.querySelector('input[name="PreRdiafather"]:checked')?.value || '0'),
        PreRdiamother: parseInt(document.querySelector('input[name="PreRdiamother"]:checked')?.value || '0'),
        PreRdiabrother: parseInt(document.querySelector('input[name="PreRdiabrother"]:checked')?.value || '0'),
        PreRdiasister: parseInt(document.querySelector('input[name="PreRdiasister"]:checked')?.value || '0'),
        current_smoking: parseInt(document.querySelector('input[name="current_smoking"]:checked')?.value || '0'),
        current_alcohol: parseInt(document.querySelector('input[name="current_alcohol"]:checked')?.value || '0'),
        PreRsleepquality: parseFloat(document.getElementById('PreRsleepquality').value),
        PreRmildactivityduration: parseFloat(document.getElementById('PreRmildactivityduration').value),
        PreRmoderate: parseFloat(document.getElementById('PreRmoderate').value),
        PreRmoderateduration: parseFloat(document.getElementById('PreRmoderateduration').value),
        PreRvigorous: parseFloat(document.getElementById('PreRvigorous').value),
        PreRvigorousduration: parseFloat(document.getElementById('PreRvigorousduration').value),
        PreRskipbreakfast: parseFloat(document.getElementById('PreRskipbreakfast').value),
        PreRlessfruit: parseFloat(document.getElementById('PreRlessfruit').value),
        PreRlessvegetable: parseFloat(document.getElementById('PreRlessvegetable').value),
        PreRmilk: parseFloat(document.getElementById('PreRmilk').value),
        PreRmeat: parseFloat(document.getElementById('PreRmeat').value),
        PreRfriedfood: parseFloat(document.getElementById('PreRfriedfood').value),
        PreRsweet: parseFloat(document.getElementById('PreRsweet').value),
        PreRwaist: parseFloat(document.getElementById('PreRwaist').value),
        PreRBMI: parseFloat(document.getElementById('PreRBMI').value),
        PreRsystolicfirst: parseFloat(document.getElementById('PreRsystolicfirst').value),
        PreRdiastolicfirst: parseFloat(document.getElementById('PreRdiastolicfirst').value),
        PreBLPPBS: parseFloat(document.getElementById('PreBLPPBS').value),
        PreBLFBS: parseFloat(document.getElementById('PreBLFBS').value),
        PreBLHBA1C: parseFloat(document.getElementById('PreBLHBA1C').value),
        PreBLCHOLESTEROL: parseFloat(document.getElementById('PreBLCHOLESTEROL').value),
        PreBLTRIGLYCERIDES: parseFloat(document.getElementById('PreBLTRIGLYCERIDES').value),
        Diabetic_Duration: parseFloat(document.getElementById('Diabetic_Duration').value),
        PostRgroupname: parseInt(document.getElementById('PostRgroupname').value || '0'),
    };
}

function validateFormData(data) {
    // Check critical fields
    if (!data.PreBLGender) { showNotification('Please select a gender.', 'warning'); return false; }
    if (isNaN(data.PostBLAge) || data.PostBLAge < 18) { showNotification('Please enter a valid age (18-90).', 'warning'); return false; }
    if (!data.PreRarea) { showNotification('Please select place of residence.', 'warning'); return false; }
    if (!data.PostRgroupname) { showNotification('Please select a care plan.', 'warning'); return false; }
    if (isNaN(data.PreBLHBA1C) || data.PreBLHBA1C <= 0) { showNotification('Please enter pre-treatment HbA1c.', 'warning'); return false; }
    if (isNaN(data.PreBLFBS) || data.PreBLFBS <= 0) { showNotification('Please enter fasting blood sugar.', 'warning'); return false; }
    if (isNaN(data.PreRBMI) || data.PreRBMI <= 0) { showNotification('Please enter BMI.', 'warning'); return false; }
    return true;
}

async function fetchPrediction(data) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return response.json();
}

async function fetchExplanation(data) {
    const response = await fetch(`${API_BASE_URL}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return response.json();
}

function displayResults(prediction, explanation) {
    setTimeout(() => {
        showLoading(false);
        elements.assessmentForm.classList.add('hidden');
        elements.resultsSection.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });

        setTimeout(() => {
            animateRiskScore(prediction);
            displayClinicalInterpretation(prediction);
            displayFactors(explanation.top_contributing_factors);
            displayRecommendations(prediction.risk_level);
            initSimulator();
            showChatWidget();
        }, 300);
    }, 1000);
}

/**
 * Animate the HbA1c score display
 * Maps HbA1c value (3-16%) to the ring/meter
 */
function animateRiskScore(prediction) {
    const hba1c = prediction.predicted_hba1c;
    const riskLevel = prediction.risk_level;

    // Map risk level to CSS class
    let cssClass = 'low';
    if (riskLevel === 'HIGH_RISK') cssClass = 'high';
    else if (riskLevel === 'DIABETIC') cssClass = 'medium';
    else if (riskLevel === 'PRE_DIABETIC') cssClass = 'medium';

    elements.riskCard.className = 'risk-card ' + cssClass;
    elements.riskConfidence.textContent = prediction.confidence;

    // Animate HbA1c value counter
    animateCounter(elements.riskPercentage, 0, hba1c, 1500);

    // Progress ring: map HbA1c 3-16 to 0-100%
    const normalizedPct = Math.min(((hba1c - 3) / 13) * 100, 100);
    const circumference = 2 * Math.PI * 54;
    const offset = circumference - (normalizedPct / 100) * circumference;

    setTimeout(() => {
        elements.progressRing.style.strokeDashoffset = offset;
    }, 100);

    // Risk level label
    const labelMap = {
        'NORMAL': 'NORMOGLYCEMIA',
        'PRE_DIABETIC': 'PREDIABETES',
        'DIABETIC': 'DIABETES',
        'HIGH_RISK': 'HIGH RISK'
    };
    setTimeout(() => {
        elements.riskLevel.textContent = labelMap[riskLevel] || riskLevel;
    }, 500);

    // Meter pointer
    setTimeout(() => {
        elements.meterPointer.style.left = `${normalizedPct}%`;
    }, 100);
}

/**
 * Clinical interpretation from reference sheet:
 * outcome_line, response_line, target_line
 */
function displayClinicalInterpretation(prediction) {
    const postHbA1c = prediction.predicted_hba1c;
    const preHbA1c = currentFormData?.PreBLHBA1C || 0;
    const age = currentFormData?.PostBLAge || 0;

    function getCategory(v) {
        if (v < 5.7) return 'Normoglycemia';
        if (v <= 6.4) return 'Prediabetes';
        return 'Diabetes';
    }

    const preCat = getCategory(preHbA1c);
    const postCat = getCategory(postHbA1c);
    const order = { 'Normoglycemia': 0, 'Prediabetes': 1, 'Diabetes': 2 };

    let traj = 'Persistence';
    if (order[postCat] < order[preCat]) traj = 'Regression';
    else if (order[postCat] > order[preCat]) traj = 'Progression';

    const outcomeLine = `Predicted outcome: ${preCat} → ${postCat} (${traj})`;

    // Response classification (baseline Diabetes only)
    const delta = preHbA1c - postHbA1c; // positive = improvement
    let responseLine = '';
    if (preCat === 'Diabetes') {
        if (delta >= 1.0) responseLine = `Predicted response: Major improvement – Risk reduction achieved (ΔHbA1c ${delta >= 0 ? '+' : ''}${delta.toFixed(2)}%)`;
        else if (delta >= 0.5) responseLine = `Predicted response: Clinically meaningful improvement (ΔHbA1c +${delta.toFixed(2)}%)`;
        else if (delta >= 0) responseLine = `Predicted response: Stabilization / modest improvement (ΔHbA1c +${delta.toFixed(2)}%)`;
        else responseLine = `Predicted response: Non-response (ΔHbA1c ${delta.toFixed(2)}%)`;
    }

    // Target achievement
    let targetLine = '';
    if (preCat === 'Diabetes' && preHbA1c > 7.0) {
        const target = age < 65 ? 7.0 : 7.5;
        const achieved = postHbA1c <= target ? 'Achieved ✅' : 'Not achieved ❌';
        targetLine = `Glycemic control target: ≤${target}% (age ${age}) | ${achieved}`;
    }

    // Update DOM
    const outEl = document.getElementById('outcomeLine');
    const resEl = document.getElementById('responseLine');
    const tgtEl = document.getElementById('targetLine');

    if (outEl) outEl.textContent = outcomeLine;
    if (resEl) resEl.textContent = responseLine;
    if (tgtEl) tgtEl.textContent = targetLine;
}

function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    const diff = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = start + diff * easeOut;
        element.textContent = current.toFixed(1);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function displayFactors(factors) {
    elements.factorsContainer.innerHTML = '';
    const maxImpact = Math.max(...factors.map(f => Math.abs(f.impact)));

    factors.forEach((factor, index) => {
        const card = createFactorCard(factor, maxImpact);
        elements.factorsContainer.appendChild(card);
        setTimeout(() => { card.classList.add('animate'); }, 50);
    });
}

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

    setTimeout(() => {
        const bar = card.querySelector('.factor-bar-fill');
        bar.style.width = `${normalizedImpact}%`;
    }, 300);

    return card;
}

/**
 * Format encoded feature names to human-readable labels
 */
function formatFeatureName(name) {
    const nameMap = {
        'PostBLAge': 'Age',
        'PreRwaist': 'Waist Circumference',
        'PreRBMI': 'BMI (Body Mass Index)',
        'PreRsystolicfirst': 'Systolic Blood Pressure',
        'PreRdiastolicfirst': 'Diastolic Blood Pressure',
        'PreBLPPBS': 'Post-Prandial Blood Sugar',
        'PreBLFBS': 'Fasting Blood Sugar',
        'PreBLHBA1C': 'Pre-Treatment HbA1c',
        'PreBLCHOLESTEROL': 'Cholesterol',
        'PreBLTRIGLYCERIDES': 'Triglycerides',
        'Diabetic_Duration': 'Diabetes Duration',
        'PreRmildactivityduration': 'Mild Activity Duration',
        'PreRmoderateduration': 'Moderate Activity Duration',
        'PreRvigorousduration': 'Vigorous Activity Duration',
        // One-hot encoded categorical features
        'PreBLGender_Male': 'Gender: Male',
        'PreBLGender_Female': 'Gender: Female',
        'PreRarea_1': 'Urban Area',
        'PreRarea_2': 'Rural Area',
        'PreRarea_1.0': 'Urban Area',
        'PreRarea_2.0': 'Rural Area',
        'PreRdiafather_0': 'Father: No Diabetes',
        'PreRdiafather_1': 'Father: Has Diabetes',
        'PreRdiafather_0.0': 'Father: No Diabetes',
        'PreRdiafather_1.0': 'Father: Has Diabetes',
        'PreRdiamother_0': 'Mother: No Diabetes',
        'PreRdiamother_1': 'Mother: Has Diabetes',
        'PreRdiamother_0.0': 'Mother: No Diabetes',
        'PreRdiamother_1.0': 'Mother: Has Diabetes',
        'PreRdiabrother_0': 'Brother: No Diabetes',
        'PreRdiabrother_1': 'Brother: Has Diabetes',
        'PreRdiabrother_0.0': 'Brother: No Diabetes',
        'PreRdiabrother_1.0': 'Brother: Has Diabetes',
        'PreRdiasister_0': 'Sister: No Diabetes',
        'PreRdiasister_1': 'Sister: Has Diabetes',
        'PreRdiasister_0.0': 'Sister: No Diabetes',
        'PreRdiasister_1.0': 'Sister: Has Diabetes',
        'current_smoking_0': 'Non-Smoker',
        'current_smoking_1': 'Current Smoker',
        'current_smoking_0.0': 'Non-Smoker',
        'current_smoking_1.0': 'Current Smoker',
        'current_alcohol_0': 'No Alcohol',
        'current_alcohol_1': 'Current Drinker',
        'current_alcohol_0.0': 'No Alcohol',
        'current_alcohol_1.0': 'Current Drinker',
        'PostRgroupname_1': 'Yoga Group',
        'PostRgroupname_2': 'Control Group',
        'PostRgroupname_1.0': 'Yoga Group',
        'PostRgroupname_2.0': 'Control Group',
    };

    if (nameMap[name]) return nameMap[name];

    // Marital status codes
    const maritalMap = { '1': 'Married', '1.0': 'Married', '2': 'Unmarried', '2.0': 'Unmarried', '3': 'Divorcee/Separated', '3.0': 'Divorcee/Separated', '4': 'Widow/Widower', '4.0': 'Widow/Widower', '5': 'Others', '5.0': 'Others' };
    if (name.startsWith('PreRmaritalstatus_')) { const v = name.split('_').pop(); return `Marital: ${maritalMap[v] || v}`; }

    // Education codes
    const eduMap = { '1': 'No formal schooling', '1.0': 'No formal schooling', '2': 'Primary school', '2.0': 'Primary school', '3': 'High school', '3.0': 'High school', '4': 'Intermediate', '4.0': 'Intermediate', '5': 'University', '5.0': 'University', '6': 'Univ. completed+', '6.0': 'Univ. completed+', '7': 'Others', '7.0': 'Others' };
    if (name.startsWith('PreReducation_')) { const v = name.split('_').pop(); return `Education: ${eduMap[v] || v}`; }

    // Occupation codes
    const occMap = { '1': 'Professional/Executive', '1.0': 'Professional/Executive', '2': 'Clerical/Medium business', '2.0': 'Clerical/Medium business', '3': 'Self-employed/Skilled', '3.0': 'Self-employed/Skilled', '4': 'Unskilled laborer', '4.0': 'Unskilled laborer', '5': 'Homemaker', '5.0': 'Homemaker', '6': 'Retired', '6.0': 'Retired', '7': 'Unemployed (able)', '7.0': 'Unemployed (able)', '8': 'Unemployed (unable)', '8.0': 'Unemployed (unable)', '9': 'Others', '9.0': 'Others' };
    if (name.startsWith('PreRpresentoccupation_')) { const v = name.split('_').pop(); return `Occupation: ${occMap[v] || v}`; }

    // Sleep quality codes
    const sleepMap = { '1': 'Very good', '1.0': 'Very good', '2': 'Fairly good', '2.0': 'Fairly good', '3': 'Fairly bad', '3.0': 'Fairly bad', '4': 'Very bad', '4.0': 'Very bad' };
    if (name.startsWith('PreRsleepquality_')) { const v = name.split('_').pop(); return `Sleep: ${sleepMap[v] || v}`; }

    // Activity frequency codes
    const freqMap = { '0': 'None', '0.0': 'None', '1': 'Once/month', '1.0': 'Once/month', '2': '2-3×/month', '2.0': '2-3×/month', '3': 'Once/week', '3.0': 'Once/week', '4': '2-3×/week', '4.0': '2-3×/week', '5': '4-5×/week', '5.0': '4-5×/week', '6': 'Every day', '6.0': 'Every day' };
    if (name.startsWith('PreRmoderate_')) { const v = name.split('_').pop(); return `Moderate Activity: ${freqMap[v] || v}`; }
    if (name.startsWith('PreRvigorous_')) { const v = name.split('_').pop(); return `Vigorous Activity: ${freqMap[v] || v}`; }

    // Duration codes
    const durMap = { '0': 'None', '0.0': 'None', '1': '≤10 min', '1.0': '≤10 min', '2': '10-30 min', '2.0': '10-30 min', '3': '30min-1hr', '3.0': '30min-1hr', '4': '1-1.5 hrs', '4.0': '1-1.5 hrs', '5': '>1.5 hrs', '5.0': '>1.5 hrs' };
    if (name.startsWith('PreRmildactivityduration_')) { const v = name.split('_').pop(); return `Mild Activity Duration: ${durMap[v] || v}`; }

    // Diet frequency codes (1=Usually/Often, 2=Sometimes, 3=Rarely/Never)
    const dietMap = { '1': 'Usually/Often', '1.0': 'Usually/Often', '2': 'Sometimes', '2.0': 'Sometimes', '3': 'Rarely/Never', '3.0': 'Rarely/Never' };
    if (name.startsWith('PreRskipbreakfast_')) { const v = name.split('_').pop(); return `Skip Breakfast: ${dietMap[v] || v}`; }
    if (name.startsWith('PreRlessfruit_')) { const v = name.split('_').pop(); return `Low Fruit Intake: ${dietMap[v] || v}`; }
    if (name.startsWith('PreRlessvegetable_')) { const v = name.split('_').pop(); return `Low Vegetable Intake: ${dietMap[v] || v}`; }
    if (name.startsWith('PreRmilk_')) { const v = name.split('_').pop(); return `Low Milk/Curd: ${dietMap[v] || v}`; }
    if (name.startsWith('PreRmeat_')) { const v = name.split('_').pop(); return `High Meat/Fish: ${dietMap[v] || v}`; }
    if (name.startsWith('PreRfriedfood_')) { const v = name.split('_').pop(); return `Fried Food: ${dietMap[v] || v}`; }
    if (name.startsWith('PreRsweet_')) { const v = name.split('_').pop(); return `Sweets >2×/day: ${dietMap[v] || v}`; }

    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

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

function getRecommendations(riskLevel) {
    const baseRecs = [
        { icon: '🧘', title: 'Yoga & Exercise', text: 'Regular yoga and moderate exercise can significantly improve insulin sensitivity and lower HbA1c levels.' },
        { icon: '🥗', title: 'Balanced Diet', text: 'Focus on low glycemic index foods, whole grains, vegetables, and limit refined sugars and fried foods.' }
    ];

    if (riskLevel === 'HIGH_RISK') {
        return [
            { icon: '🏥', title: 'Consult Endocrinologist', text: 'With high predicted HbA1c, schedule an appointment with a diabetes specialist for comprehensive evaluation.' },
            { icon: '📊', title: 'Monitor Daily', text: 'Regular blood glucose monitoring is critical. Track FBS and PPBS daily and maintain a health diary.' },
            { icon: '💊', title: 'Medication Review', text: 'Discuss medication adjustments with your doctor. Doses may need to be optimized for better control.' },
            ...baseRecs
        ];
    } else if (riskLevel === 'DIABETIC') {
        return [
            { icon: '📋', title: 'Regular Checkups', text: 'Schedule HbA1c tests every 3 months to monitor your diabetes management progress.' },
            { icon: '🚭', title: 'Lifestyle Changes', text: 'Quit smoking, limit alcohol, and maintain consistent sleep patterns for better glucose control.' },
            ...baseRecs
        ];
    } else if (riskLevel === 'PRE_DIABETIC') {
        return [
            { icon: '⚠️', title: 'Early Action', text: 'Pre-diabetic levels can be reversed with lifestyle changes. Act now to prevent progression to diabetes.' },
            { icon: '🏃', title: 'Increase Activity', text: 'Aim for 150 minutes of moderate activity per week. Even walking 30 minutes daily helps significantly.' },
            ...baseRecs
        ];
    } else {
        return [
            { icon: '✅', title: 'Keep It Up!', text: 'Your predicted HbA1c is in the normal range. Continue maintaining your healthy lifestyle!' },
            { icon: '🧘', title: 'Stay Active', text: 'Continue regular physical activity and yoga practice to maintain your excellent glucose control.' },
            ...baseRecs
        ];
    }
}

function showLoading(show) {
    if (show) elements.loadingOverlay.classList.add('active');
    else elements.loadingOverlay.classList.remove('active');
}

function showForm() {
    elements.resultsSection.classList.add('hidden');
    elements.assessmentForm.classList.remove('hidden');
    elements.patientForm.reset();
    resetWhatIf();
    hideChatWidget();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showNotification(message, type = 'info') {
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span class="notification-message">${message}</span>
        <button class="notification-close">&times;</button>
    `;
    notification.style.cssText = `
        position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
        background: ${type === 'error' ? '#E85D4C' : type === 'warning' ? '#FF9800' : '#2D9596'};
        color: white; padding: 12px 24px; border-radius: 8px; display: flex; align-items: center;
        gap: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 10000; max-width: 90%;
        animation: slideUp 0.3s ease;
    `;

    const style = document.createElement('style');
    style.textContent = `@keyframes slideUp { from { opacity: 0; transform: translateX(-50%) translateY(20px); } to { opacity: 1; transform: translateX(-50%) translateY(0); } }`;
    document.head.appendChild(style);
    document.body.appendChild(notification);

    notification.querySelector('.notification-close').addEventListener('click', () => notification.remove());
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(-50%) translateY(20px)';
        notification.style.transition = 'all 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function fillDemoData() {
    // Gender - radio
    document.getElementById('gender_male').checked = true;
    document.getElementById('PostBLAge').value = '55';
    // Area - radio
    document.getElementById('area_urban').checked = true;
    document.getElementById('PreRmaritalstatus').value = '1';
    document.getElementById('PreReducation').value = '4';
    document.getElementById('PreRpresentoccupation').value = '3';
    // Family history - radio groups
    document.getElementById('diafather_yes').checked = true;
    document.getElementById('diamother_no').checked = true;
    document.getElementById('diabrother_no').checked = true;
    document.getElementById('diasister_no').checked = true;
    // Smoking/alcohol - radio groups
    document.getElementById('smoking_no').checked = true;
    document.getElementById('alcohol_no').checked = true;
    document.getElementById('PreRsleepquality').value = '2';
    // Care plan - select dropdown
    document.getElementById('PostRgroupname').value = '1';
    document.getElementById('PreRmildactivityduration').value = '3';
    document.getElementById('PreRmoderate').value = '2';
    document.getElementById('PreRmoderateduration').value = '2';
    document.getElementById('PreRvigorous').value = '1';
    document.getElementById('PreRvigorousduration').value = '1';
    document.getElementById('PreRskipbreakfast').value = '2';
    document.getElementById('PreRlessfruit').value = '2';
    document.getElementById('PreRlessvegetable').value = '2';
    document.getElementById('PreRmilk').value = '2';
    document.getElementById('PreRmeat').value = '2';
    document.getElementById('PreRfriedfood').value = '2';
    document.getElementById('PreRsweet').value = '2';
    document.getElementById('PreRwaist').value = '92';
    document.getElementById('PreRBMI').value = '27.5';
    document.getElementById('PreRsystolicfirst').value = '130';
    document.getElementById('PreRdiastolicfirst').value = '84';
    document.getElementById('PreBLPPBS').value = '220';
    document.getElementById('PreBLFBS').value = '140';
    document.getElementById('PreBLHBA1C').value = '8.2';
    document.getElementById('PreBLCHOLESTEROL').value = '210';
    document.getElementById('PreBLTRIGLYCERIDES').value = '180';
    document.getElementById('Diabetic_Duration').value = '5';
    showNotification('Demo data filled! Click Predict HbA1c to see results.', 'info');
}


// =============================================
// JSON IMPORT MODAL
// =============================================

function openJsonModal() {
    const modal = document.getElementById('jsonModal');
    modal.style.display = 'flex';
    document.getElementById('jsonInput').value = '';
    document.getElementById('jsonInput').focus();
}

function closeJsonModal() {
    document.getElementById('jsonModal').style.display = 'none';
}

function applyJsonFromModal() {
    const raw = document.getElementById('jsonInput').value.trim();
    if (!raw) {
        showNotification('Please paste a JSON object first.', 'warning');
        return;
    }
    try {
        const obj = JSON.parse(raw);
        applyJsonToForm(obj);
        closeJsonModal();
        showNotification(`JSON applied! ${Object.keys(obj).length} fields loaded. Click Predict HbA1c.`, 'info');
    } catch (e) {
        showNotification('Invalid JSON. Please check the format.', 'error');
        console.error('JSON parse error:', e);
    }
}

function applyJsonToForm(json) {
    // Radio button fields: name → {valueToIdMap}
    const radioFields = {
        'PreBLGender': { 'Male': 'gender_male', 'Female': 'gender_female', 'Others': 'gender_others' },
        'PreRarea': { '1': 'area_urban', '2': 'area_rural' },
        'PreRdiafather': { '0': 'diafather_no', '1': 'diafather_yes' },
        'PreRdiamother': { '0': 'diamother_no', '1': 'diamother_yes' },
        'PreRdiabrother': { '0': 'diabrother_no', '1': 'diabrother_yes' },
        'PreRdiasister': { '0': 'diasister_no', '1': 'diasister_yes' },
        'current_smoking': { '0': 'smoking_no', '1': 'smoking_yes' },
        'current_alcohol': { '0': 'alcohol_no', '1': 'alcohol_yes' },
    };

    // Direct input/select fields (by element ID)
    const directFields = [
        'PostBLAge', 'PreRmaritalstatus', 'PreReducation', 'PreRpresentoccupation',
        'PreRsleepquality', 'PostRgroupname',
        'PreRmildactivityduration', 'PreRmoderate', 'PreRmoderateduration',
        'PreRvigorous', 'PreRvigorousduration',
        'PreRskipbreakfast', 'PreRlessfruit', 'PreRlessvegetable',
        'PreRmilk', 'PreRmeat', 'PreRfriedfood', 'PreRsweet',
        'PreRwaist', 'PreRBMI', 'PreRsystolicfirst', 'PreRdiastolicfirst',
        'PreBLPPBS', 'PreBLFBS', 'PreBLHBA1C',
        'PreBLCHOLESTEROL', 'PreBLTRIGLYCERIDES', 'Diabetic_Duration',
    ];

    for (const [key, val] of Object.entries(json)) {
        // Handle radio buttons
        if (radioFields[key]) {
            const idMap = radioFields[key];
            const targetId = idMap[String(val)];
            if (targetId) {
                document.getElementById(targetId).checked = true;
            }
            continue;
        }
        // Handle direct input/select fields
        if (directFields.includes(key)) {
            const el = document.getElementById(key);
            if (el) el.value = val;
        }
    }
}


// =============================================
// WHAT-IF SCENARIO ANALYSIS
// =============================================

async function fetchWhatIfAnalysis(formData) {
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
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const whatifData = await response.json();
        currentWhatIf = whatifData;
        elements.whatifLoading.style.display = 'none';
        displayWhatIfAnalysis(whatifData);
    } catch (error) {
        console.error('What-If API Error:', error);
        elements.whatifLoading.style.display = 'none';
        elements.whatifScenariosGrid.innerHTML = `
            <div class="whatif-error"><p>⚠️ Could not generate What-If scenarios.</p></div>
        `;
    }
}

function displayWhatIfAnalysis(data) {
    if (!data.scenarios || data.scenarios.length === 0) {
        elements.whatifScenariosGrid.innerHTML = `
            <div class="whatif-empty"><p>✅ Your current health parameters are already in healthy ranges!</p></div>
        `;
        return;
    }

    data.scenarios.forEach((scenario, index) => {
        const card = createScenarioCard(scenario);
        elements.whatifScenariosGrid.appendChild(card);
        setTimeout(() => { card.classList.add('animate'); }, 100 + index * 150);
    });

    if (data.combined_hba1c !== null && data.combined_hba1c !== undefined && data.scenarios.length > 1) {
        setTimeout(() => {
            displayCombinedOutcome(data);
        }, 100 + data.scenarios.length * 150 + 200);
    }
}

function createScenarioCard(scenario) {
    const card = document.createElement('div');
    card.className = 'whatif-card';

    const isReduction = scenario.hba1c_delta > 0;
    const deltaAbs = Math.abs(scenario.hba1c_delta).toFixed(2);
    const improvePct = Math.abs(scenario.improvement_percent).toFixed(1);
    const barWidth = Math.min(Math.abs(scenario.improvement_percent), 100);

    card.innerHTML = `
        <div class="whatif-card-icon">${scenario.icon}</div>
        <div class="whatif-card-content">
            <h4 class="whatif-card-title">${scenario.title}</h4>
            <p class="whatif-card-desc">${scenario.description}</p>
            <div class="whatif-card-comparison">
                <div class="whatif-risk-original">
                    <span class="whatif-risk-label">Current</span>
                    <span class="whatif-risk-val">${scenario.original_hba1c.toFixed(2)}</span>
                </div>
                <div class="whatif-arrow-container">
                    <svg class="whatif-arrow ${isReduction ? 'reduction' : 'increase'}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                </div>
                <div class="whatif-risk-modified">
                    <span class="whatif-risk-label">Modified</span>
                    <span class="whatif-risk-val ${isReduction ? 'improved' : 'worsened'}">${scenario.modified_hba1c.toFixed(2)}</span>
                </div>
            </div>
            <div class="whatif-delta-bar">
                <div class="whatif-delta-fill ${isReduction ? 'positive' : 'negative'}" 
                     style="width: 0%" data-width="${barWidth}%"></div>
            </div>
            <div class="whatif-delta-text ${isReduction ? 'positive' : 'negative'}">
                ${isReduction ? '↓' : '↑'} ${deltaAbs} HbA1c ${isReduction ? 'reduction' : 'increase'}
                <span class="whatif-delta-pct">(${improvePct}% ${isReduction ? 'improvement' : 'change'})</span>
            </div>
        </div>
    `;

    setTimeout(() => {
        const bar = card.querySelector('.whatif-delta-fill');
        if (bar) bar.style.width = `${barWidth}%`;
    }, 500);

    return card;
}

function displayCombinedOutcome(data) {
    elements.whatifCombinedCard.style.display = 'block';

    const originalHba1c = data.original_hba1c;
    const combinedHba1c = data.combined_hba1c;
    const totalDelta = originalHba1c - combinedHba1c;
    const totalImprovePct = originalHba1c > 0 ? (totalDelta / originalHba1c * 100) : 0;

    elements.combinedOriginalRisk.textContent = `${originalHba1c.toFixed(2)}`;
    elements.combinedModifiedRisk.textContent = `${combinedHba1c.toFixed(2)}`;

    if (combinedHba1c < originalHba1c) {
        elements.combinedModifiedRisk.classList.add('improved');
    }

    const isReduction = totalDelta > 0;
    elements.combinedDelta.innerHTML = `
        <div class="combined-delta-badge ${isReduction ? 'positive' : 'negative'}">
            ${isReduction ? '↓' : '↑'} ${Math.abs(totalDelta).toFixed(2)} total HbA1c ${isReduction ? 'reduction' : 'increase'}
            <span>(${Math.abs(totalImprovePct).toFixed(1)}% overall ${isReduction ? 'improvement' : 'change'})</span>
        </div>
        <p class="combined-risk-level">Risk Level: <strong>${data.original_risk_level}</strong> → <strong class="${(data.combined_risk_level || '').toLowerCase()}">${data.combined_risk_level || 'N/A'}</strong></p>
    `;

    setTimeout(() => { elements.whatifCombinedCard.classList.add('animate'); }, 100);
}

function resetWhatIf() {
    currentWhatIf = null;
    elements.whatifSection.style.display = 'none';
    elements.whatifScenariosGrid.innerHTML = '';
    elements.whatifCombinedCard.style.display = 'none';
    elements.whatifCombinedCard.classList.remove('animate');
    elements.combinedModifiedRisk.classList.remove('improved');
}


// =============================================
// INTERACTIVE RISK SIMULATOR
// =============================================

let simTimeout = null;

function initSimulator() {
    if (!currentFormData || !currentPrediction) return;

    const simCard = document.getElementById('interactiveSimulator');
    if (simCard) simCard.classList.remove('hidden');

    document.getElementById('simBaselineStat').textContent = currentPrediction.predicted_hba1c.toFixed(2);
    updateSimulatorUI(currentPrediction, currentPrediction, currentFormData);

    // Setup each control independently so one failure doesn't break the rest
    const controls = [
        ['Bmi', 'PreRBMI', 15, 60],
        ['Fbs', 'PreBLFBS', 50, 400],
        ['Hba1c', 'PreBLHBA1C', 3, 16],
        ['Bp', 'PreRsystolicfirst', 80, 220],
        ['Ppbs', 'PreBLPPBS', 70, 600],
        ['Chol', 'PreBLCHOLESTEROL', 80, 400],
        ['Trig', 'PreBLTRIGLYCERIDES', 50, 1000],
        ['Waist', 'PreRwaist', 50, 150],
    ];
    controls.forEach(([prefix, field, min, max]) => {
        try {
            setupSimControl(prefix, field, min, max);
        } catch (err) {
            console.error(`[Sim] Failed to setup ${prefix}:`, err);
        }
    });
}

function setupSimControl(idPrefix, fieldName, min, max) {
    const slider = document.getElementById(`sim${idPrefix}Slider`);
    const num = document.getElementById(`sim${idPrefix}Num`);

    if (!slider || !num || !currentFormData) {
        console.warn(`[Sim] Missing element or data for ${idPrefix} (field=${fieldName})`);
        return;
    }

    // Dynamically set attributes on the number input so the browser renders values
    num.setAttribute('min', min);
    num.setAttribute('max', max);
    num.setAttribute('step', slider.step || 'any');

    let val = Number(currentFormData[fieldName]);
    if (isNaN(val) || val === 0) val = min;
    if (val < min) val = min;
    if (val > max) val = max;

    slider.value = val;
    num.value = val;

    slider.oninput = (e) => { num.value = e.target.value; triggerSimulate(); };
    num.oninput = (e) => {
        let v = Number(e.target.value);
        if (!isNaN(v) && v >= min && v <= max) { slider.value = v; triggerSimulate(); }
    };
}

function triggerSimulate() {
    clearTimeout(simTimeout);
    simTimeout = setTimeout(simulateRisk, 300);
}

async function simulateRisk() {
    if (!currentFormData) return;

    const simulatedData = { ...currentFormData };

    // Helper: read num input, fall back to currentFormData value
    function readSim(elId, field) {
        const v = parseFloat(document.getElementById(elId)?.value);
        return isNaN(v) ? currentFormData[field] : v;
    }

    simulatedData.PreRBMI = readSim('simBmiNum', 'PreRBMI');
    simulatedData.PreBLFBS = readSim('simFbsNum', 'PreBLFBS');
    simulatedData.PreBLHBA1C = readSim('simHba1cNum', 'PreBLHBA1C');
    simulatedData.PreRsystolicfirst = readSim('simBpNum', 'PreRsystolicfirst');
    simulatedData.PreBLPPBS = readSim('simPpbsNum', 'PreBLPPBS');
    simulatedData.PreBLCHOLESTEROL = readSim('simCholNum', 'PreBLCHOLESTEROL');
    simulatedData.PreBLTRIGLYCERIDES = readSim('simTrigNum', 'PreBLTRIGLYCERIDES');
    simulatedData.PreRwaist = readSim('simWaistNum', 'PreRwaist');

    try {
        const response = await fetchPrediction(simulatedData);
        updateSimulatorUI(currentPrediction, response, simulatedData);
    } catch (e) {
        console.error("Simulation failed", e);
    }
}

function updateSimulatorUI(baselinePred, targetPred, simData) {
    const hba1c = targetPred.predicted_hba1c;
    const levelStr = targetPred.risk_level.replace('_', ' ').toUpperCase();

    document.getElementById('simRiskValue').textContent = hba1c.toFixed(2);
    document.getElementById('simRiskLabel').textContent = levelStr;
    document.getElementById('simTargetStat').textContent = hba1c.toFixed(2);

    const delta = (hba1c - baselinePred.predicted_hba1c).toFixed(2);
    const deltaEl = document.getElementById('simDeltaStat');
    deltaEl.textContent = (delta > 0 ? '+' : '') + delta;
    deltaEl.className = 'sim-stat-val ' + (delta > 0 ? 'negative' : (delta < 0 ? 'positive' : ''));

    // Map HbA1c 3-16 to arc
    const normalized = Math.min(((hba1c - 3) / 13) * 100, 100);
    const trackFilled = 219.91 * (1 - (normalized / 100));
    const simTrack = document.getElementById('simTrack');
    if (simTrack) {
        simTrack.style.strokeDashoffset = trackFilled;
        let color = 'var(--risk-low)';
        if (targetPred.risk_level === 'PRE_DIABETIC' || targetPred.risk_level === 'DIABETIC') color = 'var(--risk-medium)';
        if (targetPred.risk_level === 'HIGH_RISK') color = 'var(--risk-high)';
        simTrack.style.stroke = color;
    }

    // Baseline marker
    const baseNorm = Math.min(((baselinePred.predicted_hba1c - 3) / 13) * 100, 100);
    const baseAngle = 180 * (baseNorm / 100);
    const baseGroup = document.getElementById('simBaselineGroup');
    const baseTextGroup = document.getElementById('simBaselineTextGroup');
    if (baseGroup) baseGroup.style.transform = `rotate(${baseAngle}deg)`;
    if (baseTextGroup) baseTextGroup.style.transform = `translate(18px, 90px) rotate(${-baseAngle}deg)`;

    // Target marker
    const targetAngle = 180 * (normalized / 100);
    const targetGroup = document.getElementById('simTargetGroup');
    const targetTextGroup = document.getElementById('simTargetTextGroup');
    if (targetGroup) targetGroup.style.transform = `rotate(${targetAngle}deg)`;
    if (targetTextGroup) targetTextGroup.style.transform = `translate(18px, 110px) rotate(${-targetAngle}deg)`;
}


// =============================================
// AI CHAT FUNCTIONALITY
// =============================================

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
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
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

    if (chatIsOpen && !chatInitialized) initializeChat();
    if (chatIsOpen) chatElems.input.focus();
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
        addChatMessage('system', '⚠️ Could not connect to DiabeSense AI. Make sure NVIDIA_API_KEY is set and agno is installed.');
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
            body: JSON.stringify({ session_id: chatSessionId, message: message })
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
    indicator.innerHTML = `<div class="chat-bubble typing-indicator"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
    chatElems.messages.appendChild(indicator);
    chatElems.messages.scrollTop = chatElems.messages.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById('chatTyping');
    if (el) el.remove();
}

function setChatStatus(text) {
    if (chatElems.status) chatElems.status.innerHTML = `<span class="status-dot"></span> ${text}`;
}

function formatChatMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/^\s*[-•]\s+(.+)/gm, '<li>$1</li>')
        .replace(/^\s*(\d+)\.\s+(.+)/gm, '<li>$2</li>')
        .replace(/\n/g, '<br>');
}

document.addEventListener('DOMContentLoaded', init);
window.fillDemoData = fillDemoData;
