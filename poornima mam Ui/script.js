/**
 * DiabSense+ Frontend Application
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
    restoreFormFromStorage();
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

        saveFormToStorage(formData);
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
        PreRsleepquality: Number(document.getElementById('PreRsleepquality').value),
        PreRmildactivity: Number(document.getElementById('PreRmildactivity').value),
        PreRmildactivityduration: Number(document.getElementById('PreRmildactivityduration').value),
        PreRmoderate: Number(document.getElementById('PreRmoderate').value),
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
    const err = (msg) => { showNotification(msg, 'warning'); return false; };
    const dropSel = (id, label) => {
        const v = document.getElementById(id)?.value;
        if (!v || v === '') return err(`Please select ${label}.`);
        return true;
    };
    const inRange = (val, min, max, label) => {
        if (isNaN(val) || val < min || val > max)
            return err(`${label} must be between ${min} and ${max}.`);
        return true;
    };

    // Demographics
    if (!data.PreBLGender) return err('Please select a gender.');
    if (!inRange(data.PostBLAge, 18, 90, 'Age')) return false;
    if (!data.PreRarea) return err('Please select place of residence.');
    if (!dropSel('PreRmaritalstatus', 'marital status')) return false;
    if (!dropSel('PreReducation', 'education level')) return false;
    if (!dropSel('PreRpresentoccupation', 'occupation')) return false;
    // Lifestyle
    if (!dropSel('PreRsleepquality', 'sleep quality')) return false;
    if (!dropSel('PostRgroupname', 'a care plan')) return false;
    // Physical Activity
    if (!dropSel('PreRmildactivity', 'mild activity frequency')) return false;
    if (!dropSel('PreRmildactivityduration', 'mild activity duration')) return false;
    if (!dropSel('PreRmoderate', 'moderate activity frequency')) return false;
    if (!dropSel('PreRmoderateduration', 'moderate activity duration')) return false;
    if (!dropSel('PreRvigorous', 'vigorous activity frequency')) return false;
    if (!dropSel('PreRvigorousduration', 'vigorous activity duration')) return false;
    // Diet
    if (!dropSel('PreRskipbreakfast', 'breakfast habit')) return false;
    if (!dropSel('PreRlessfruit', 'fruit intake')) return false;
    if (!dropSel('PreRlessvegetable', 'vegetable intake')) return false;
    if (!dropSel('PreRmilk', 'milk/curd intake')) return false;
    if (!dropSel('PreRmeat', 'meat/fish intake')) return false;
    if (!dropSel('PreRfriedfood', 'fried food intake')) return false;
    if (!dropSel('PreRsweet', 'sweet intake')) return false;
    // Measurements
    if (!inRange(data.PreRwaist, 50, 150, 'Waist circumference (cm)')) return false;
    if (!inRange(data.PreRBMI, 10, 60, 'BMI')) return false;
    if (!inRange(data.PreRsystolicfirst, 70, 250, 'Systolic BP (mmHg)')) return false;
    if (!inRange(data.PreRdiastolicfirst, 40, 150, 'Diastolic BP (mmHg)')) return false;
    // Blood work
    if (!inRange(data.PreBLPPBS, 70, 600, 'PPBS (mg/dL)')) return false;
    if (!inRange(data.PreBLFBS, 50, 400, 'FBS (mg/dL)')) return false;
    if (!inRange(data.PreBLHBA1C, 4.0, 18.0, 'HbA1c (%)')) return false;
    if (!inRange(data.PreBLCHOLESTEROL, 80, 400, 'Cholesterol (mg/dL)')) return false;
    if (!inRange(data.PreBLTRIGLYCERIDES, 50, 1000, 'Triglycerides (mg/dL)')) return false;
    if (!inRange(data.Diabetic_Duration, 0, 60, 'Diabetic duration (years)')) return false;
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
            displayInputSummary(currentFormData);
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

    // HbA1c change summary line
    const delta = preHbA1c - postHbA1c; // positive = improvement
    let deltaLine = '';
    if (preHbA1c > 0) {
        const absDelta = Math.abs(delta).toFixed(1);
        if (delta > 0) {
            deltaLine = `HbA1c ${preHbA1c.toFixed(1)}% → ${postHbA1c.toFixed(1)}% ↓ ${absDelta}% HbA1c reduction`;
        } else if (delta < 0) {
            deltaLine = `HbA1c ${preHbA1c.toFixed(1)}% → ${postHbA1c.toFixed(1)}% ↑ ${absDelta}% HbA1c increase`;
        } else {
            deltaLine = `HbA1c ${preHbA1c.toFixed(1)}% → ${postHbA1c.toFixed(1)}% (No change)`;
        }
    }

    // Response classification — no delta values in text (per user spec)
    let responseLine = '';
    if (preCat === 'Diabetes') {
        if (delta >= 1.0) responseLine = 'Predicted response: Major improvement – Risk reduction achieved (≥1.0% reduction)';
        else if (delta >= 0.5) responseLine = 'Predicted response: Clinically meaningful improvement (≥0.5% reduction)';
        else if (delta >= 0) responseLine = 'Predicted response: Stabilization / modest improvement';
        else responseLine = 'Predicted response: Non-response (increase in HbA1c)';
    }

    // Target achievement
    let targetLine = '';
    if (preCat === 'Diabetes' && preHbA1c > 7.0) {
        const ageDisplay = Math.round(age);
        const target = age < 65 ? 7.0 : 7.5;
        const baselineVal = Math.round(preHbA1c * 10) / 10;
        const postVal = Math.round(postHbA1c * 10) / 10;
        const baselineAtTarget = baselineVal <= target;
        const postAtTarget = postVal <= target;
        let achieved;
        if (baselineAtTarget) {
            achieved = postAtTarget ? 'At target ✅' : 'Above target ❌';
        } else {
            achieved = postAtTarget ? 'Achieved target ✅' : 'Not achieved ❌';
        }
        targetLine = `Glycemic Control target (HbA1c) : ≤${target.toFixed(1)}% (age ${ageDisplay}) | ${achieved}`;
    }

    // Update DOM
    const outEl = document.getElementById('outcomeLine');
    const resEl = document.getElementById('responseLine');
    const tgtEl = document.getElementById('targetLine');
    const deltaEl = document.getElementById('hba1cDeltaLine');

    if (deltaEl) { deltaEl.textContent = deltaLine; deltaEl.style.display = deltaLine ? '' : 'none'; }
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

    // Filter out any group_x_ cross features that slipped through backend
    let filtered = factors.filter(f => !f.feature.includes('group_x_'));
    if (filtered.length === 0) return;

    // Deduplicate factors mapping to the same base feature (keep the one with largest absolute impact)
    const dedupedMap = new Map();
    filtered.forEach(factor => {
        const base = factor.feature.replace(/_\d+(\.\d+)?$/, '');
        const currentAbs = Math.abs(factor.impact);
        if (!dedupedMap.has(base) || Math.abs(dedupedMap.get(base).impact) < currentAbs) {
            dedupedMap.set(base, factor);
        }
    });
    filtered = Array.from(dedupedMap.values());

    // Guarantee Intervention (PostRgroupname) appears in top 3
    const interventionIdx = filtered.findIndex(f => f.feature.startsWith('PostRgroupname') || f.feature === 'PostRgroupname');
    if (interventionIdx >= 0 && interventionIdx >= 3) {
        // Move it to position 2 (3rd slot)
        const [item] = filtered.splice(interventionIdx, 1);
        filtered.splice(2, 0, item);
    } else if (interventionIdx < 0) {
        // Not present at all — inject a synthetic entry based on care plan chosen
        const isYoga = currentFormData && currentFormData.PostRgroupname === 1;
        filtered.splice(Math.min(2, filtered.length), 0, {
            feature: 'PostRgroupname',
            impact: isYoga ? -0.05 : 0.05,
            direction: isYoga ? 'Reduces HbA1c' : 'Increases HbA1c',
            interpretation: isYoga
                ? 'Yoga intervention slightly decreases predicted HbA1c'
                : 'Standard care (without Yoga) slightly increases predicted HbA1c'
        });
    }

    const maxImpact = Math.max(...filtered.map(f => Math.abs(f.impact)));
    let anyRedFlag = false; // NEW: Track if any red flag was generated

    filtered.forEach((factor, index) => {
        // Pass currentFormData here
        const card = createFactorCard(factor, maxImpact, currentFormData);
        if (card.dataset.hasRedFlag === 'true') anyRedFlag = true; // Check flag
        elements.factorsContainer.appendChild(card);
        setTimeout(() => { card.classList.add('animate'); }, 50);
    });

    // NEW: Append legend at the bottom ONLY if a red flag is present (Audio 1 Request)
    if (anyRedFlag) {
        const legend = document.createElement('div');
        legend.className = 'red-flag-legend';
        legend.innerHTML = `<strong>🚩 Red Flag Notice:</strong> Variables marked with a red flag indicate that the AI model's interpretation goes against usual clinical expectations for your specific data. This does not mean the habit is healthy or lowers HbA1c.`;
        legend.style.cssText = "margin-top: 20px; padding: 12px; background: rgba(232, 93, 76, 0.08); border-left: 4px solid #E85D4C; border-radius: 4px; color: #3c1e21; font-size: 0.85rem; line-height: 1.5;";
        elements.factorsContainer.appendChild(legend);
    }
}

function getFormattedInputValue(feature, value) {
    if (value === undefined || value === null || value === '') return '';

    const freqMap = { 0: 'None', 1: 'Once/month', 2: '2-3×/month', 3: 'Once/week', 4: '2-3×/week', 5: '4-5×/week', 6: 'Every day' };
    const durMap = { 0: 'None', 1: '≤10 min', 2: '10-30 min', 3: '30min-1hr', 4: '1-1.5hrs', 5: '>1.5hrs' };
    const dietMap = { 1: 'Usually/Often', 2: 'Sometimes', 3: 'Rarely/Never' };
    const maritalMap = { 1: 'Married', 2: 'Unmarried', 3: 'Divorcee/Separated', 4: 'Widow/Widower', 5: 'Others' };
    const eduMap = { 1: 'No schooling', 2: 'Primary', 3: 'High school', 4: 'Intermediate', 5: 'University', 6: 'Univ+', 7: 'Others' };
    const occMap = { 1: 'Professional', 2: 'Clerical', 3: 'Self-employed', 4: 'Unskilled', 5: 'Homemaker', 6: 'Retired', 7: 'Unemployed(able)', 8: 'Unemployed(unable)', 9: 'Others' };
    const sleepMap = { 1: 'Very good', 2: 'Fairly good', 3: 'Fairly bad', 4: 'Very bad' };
    const careMap = { 1: 'Standard + Yoga', 2: 'Standard care' };

    switch (feature) {
        case 'PostBLAge': return `${value} yrs`;
        case 'PreBLGender': return value;
        case 'PreRarea': return value === 1 ? 'Urban' : 'Rural';
        case 'PreRmaritalstatus': return maritalMap[value] || value;
        case 'PreReducation': return eduMap[value] || value;
        case 'PreRpresentoccupation': return occMap[value] || value;
        case 'PreRdiafather':
        case 'PreRdiamother':
        case 'PreRdiabrother':
        case 'PreRdiasister':
        case 'current_smoking':
        case 'current_alcohol': return value === 1 ? 'Yes' : 'No';
        case 'PreRsleepquality': return sleepMap[value] || value;
        case 'PostRgroupname': return careMap[value] || value;
        case 'PreRmildactivity':
        case 'PreRmoderate':
        case 'PreRvigorous': return freqMap[value] || value;
        case 'PreRmildactivityduration':
        case 'PreRmoderateduration':
        case 'PreRvigorousduration': return durMap[value] || value;
        case 'PreRskipbreakfast':
        case 'PreRlessfruit':
        case 'PreRlessvegetable':
        case 'PreRmilk':
        case 'PreRmeat':
        case 'PreRfriedfood':
        case 'PreRsweet': return dietMap[value] || value;
        case 'PreRwaist': return `${value} cm`;
        case 'PreRBMI': return `${value} kg/m²`;
        case 'PreRsystolicfirst':
        case 'PreRdiastolicfirst': return `${value} mmHg`;
        case 'PreBLPPBS':
        case 'PreBLFBS':
        case 'PreBLCHOLESTEROL':
        case 'PreBLTRIGLYCERIDES': return `${value} mg/dL`;
        case 'PreBLHBA1C': return `${value}%`;
        case 'Diabetic_Duration': return `${value} yrs`;
        default: return value;
    }
}

const MODIFIABLE_VARS = [
    {
        name: 'PreBLTRIGLYCERIDES',
        title: 'Manage Triglycerides',
        icon: '🩸',
        desc: (orig, sug) => `What if your triglyceride level was ${sug} instead of ${orig}?`,
        changeTemplate: (orig, sug) => `Triglycerides: ${orig} → ${sug}`,
        isUnfavourable: (v) => v >= 150,
        getNextValue: (v) => v - 10,
        reachedTarget: (v) => v < 150
    },
    {
        name: 'PreBLCHOLESTEROL',
        title: 'Lower Total Cholesterol',
        icon: '🧪',
        desc: (orig, sug) => `What if your total cholesterol was ${sug} instead of ${orig}?`,
        changeTemplate: (orig, sug) => `Cholesterol: ${orig} → ${sug}`,
        isUnfavourable: (v) => v >= 200,
        getNextValue: (v) => v - 10,
        reachedTarget: (v) => v < 200
    },
    {
        name: 'PreRsystolicfirst',
        title: 'Control Systolic BP',
        icon: '❤️',
        desc: (orig, sug) => `What if your systolic blood pressure was ${sug} instead of ${orig}?`,
        changeTemplate: (orig, sug) => `Systolic BP: ${orig} → ${sug}`,
        isUnfavourable: (v) => v >= 130,
        getNextValue: (v) => v - 5,
        reachedTarget: (v) => v < 120
    },
    {
        name: 'PreRdiastolicfirst',
        title: 'Control Diastolic BP',
        icon: '💓',
        desc: (orig, sug) => `What if your diastolic blood pressure was ${sug} instead of ${orig}?`,
        changeTemplate: (orig, sug) => `Diastolic BP: ${orig} → ${sug}`,
        isUnfavourable: (v) => v >= 80,
        getNextValue: (v) => v - 5,
        reachedTarget: (v) => v < 80
    },
    {
        name: 'PreRBMI',
        title: 'Optimize Body Weight (BMI)',
        icon: '⚖️',
        desc: (orig, sug) => `What if your BMI was ${sug} instead of ${orig}?`,
        changeTemplate: (orig, sug) => `BMI: ${orig} → ${sug}`,
        isUnfavourable: (v) => v >= 25,
        getNextValue: (v) => Math.max(v - 0.5, 18.5),
        reachedTarget: (v) => v < 25
    },
    {
        name: 'PreRwaist',
        title: 'Reduce Waist Circumference',
        icon: '📏',
        desc: (orig, sug) => `What if your waist circumference was ${sug} instead of ${orig}?`,
        changeTemplate: (orig, sug) => `Waist: ${orig} → ${sug}`,
        isUnfavourable: (v, gender) => v >= (gender === 'Female' ? 80 : 90),
        getNextValue: (v) => v - 2,
        reachedTarget: (v, gender) => v < (gender === 'Female' ? 80 : 90)
    },
    {
        name: 'PreRsleepquality',
        title: 'Improve Sleep Quality',
        icon: '😴',
        desc: (orig, sug) => `What if your sleep quality was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Sleep Quality: ${orig} → ${sug}`,
        isUnfavourable: (v) => v >= 3,
        getNextValue: (v) => v - 1,
        reachedTarget: (v) => v <= 2
    },
    {
        name: 'PreRmildactivityduration',
        title: 'Increase Mild Activity Duration',
        icon: '🚶',
        desc: (orig, sug) => `What if your mild activity duration was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Mild Activity Duration: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 3,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 3
    },
    {
        name: 'PreRmoderate',
        title: 'Increase Moderate Activity Frequency',
        icon: '🏃',
        desc: (orig, sug) => `What if your moderate activity frequency was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Moderate Activity Frequency: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 5,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 5
    },
    {
        name: 'PreRmoderateduration',
        title: 'Increase Moderate Activity Duration',
        icon: '⏱️',
        desc: (orig, sug) => `What if your moderate activity duration was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Moderate Activity Duration: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 3,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 3
    },
    {
        name: 'PreRvigorous',
        title: 'Increase Vigorous Activity Frequency',
        icon: '⚡',
        desc: (orig, sug) => `What if your vigorous activity frequency was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Vigorous Activity Frequency: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 4,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 4
    },
    {
        name: 'PreRvigorousduration',
        title: 'Increase Vigorous Activity Duration',
        icon: '💪',
        desc: (orig, sug) => `What if your vigorous activity duration was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Vigorous Activity Duration: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 3,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 3
    },
    {
        name: 'current_smoking',
        title: 'Quit Smoking',
        icon: '🚭',
        desc: (orig, sug) => `What if your smoking status was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Smoking: ${orig} → ${sug}`,
        isUnfavourable: (v) => v === 1,
        getNextValue: (v) => 0,
        reachedTarget: (v) => v === 0
    },
    {
        name: 'current_alcohol',
        title: 'Limit Alcohol Intake',
        icon: '🍷',
        desc: (orig, sug) => `What if your alcohol consumption was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Alcohol: ${orig} → ${sug}`,
        isUnfavourable: (v) => v === 1,
        getNextValue: (v) => 0,
        reachedTarget: (v) => v === 0
    },
    {
        name: 'PreRskipbreakfast',
        title: 'Eat Regular Breakfast',
        icon: '🍳',
        desc: (orig, sug) => `What if your breakfast habit was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Breakfast Habit: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 3,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 3
    },
    {
        name: 'PreRlessfruit',
        title: 'Increase Fruit Consumption',
        icon: '🍎',
        desc: (orig, sug) => `What if your fruit intake was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Fruit Intake: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 3,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 3
    },
    {
        name: 'PreRlessvegetable',
        title: 'Increase Vegetable Consumption',
        icon: '🥦',
        desc: (orig, sug) => `What if your vegetable intake was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Vegetable Intake: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 3,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 3
    },
    {
        name: 'PreRmilk',
        title: 'Increase Milk/Curd Intake',
        icon: '🥛',
        desc: (orig, sug) => `What if your milk/curd intake was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Milk/Curd Intake: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 3,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 3
    },
    {
        name: 'PreRfriedfood',
        title: 'Reduce Fried Food Intake',
        icon: '🍟',
        desc: (orig, sug) => `What if your fried food intake was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Fried Food: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 3,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 3
    },
    {
        name: 'PreRsweet',
        title: 'Reduce Sweet Intake',
        icon: '🍬',
        desc: (orig, sug) => `What if your sweet intake was "${sug}" instead of "${orig}"?`,
        changeTemplate: (orig, sug) => `Sweet Intake: ${orig} → ${sug}`,
        isUnfavourable: (v) => v < 3,
        getNextValue: (v) => v + 1,
        reachedTarget: (v) => v >= 3
    }
];

function getRiskLevelString(hba1c) {
    if (hba1c < 5.7) return 'NORMAL';
    if (hba1c < 6.5) return 'PRE_DIABETIC';
    if (hba1c < 8.0) return 'DIABETIC';
    return 'HIGH_RISK';
}

async function simulateScenario(variable, formData, baselineHba1c) {
    let currentValue = formData[variable.name];
    let gender = formData.PreBLGender;
    let nextValue = currentValue;
    let iteration = 0;
    const maxIterations = 100;
    const FLOAT_TOLERANCE = 1e-8;

    let modifiedData = { ...formData };

    while (iteration < maxIterations) {
        if (variable.reachedTarget(nextValue, gender)) {
            break;
        }

        let updatedValue = variable.getNextValue(nextValue);
        if (updatedValue === nextValue) {
            break;
        }
        nextValue = updatedValue;
        iteration++;

        modifiedData[variable.name] = nextValue;

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(modifiedData)
            });
            if (!response.ok) continue;

            const predResult = await response.json();
            const predictedVal = predResult.predicted_hba1c;
            const reduction = baselineHba1c - predictedVal;

            console.log(`[What-If Sim] Var: ${variable.name}, Val: ${nextValue}, Pred: ${predictedVal.toFixed(4)}, Reduction: ${reduction.toFixed(4)}`);

            if (reduction >= 0.10 - FLOAT_TOLERANCE) {
                const origLabel = getFormattedInputValue(variable.name, currentValue);
                const sugLabel = getFormattedInputValue(variable.name, nextValue);

                return {
                    scenario_id: 0,
                    title: variable.title,
                    description: typeof variable.desc === 'function' ? variable.desc(origLabel, sugLabel) : variable.desc,
                    change_summary: variable.changeTemplate(origLabel, sugLabel),
                    original_hba1c: baselineHba1c,
                    modified_hba1c: predictedVal,
                    hba1c_delta: Number(reduction.toFixed(2)),
                    improvement_percent: Number(((reduction / baselineHba1c) * 100).toFixed(2)),
                    icon: variable.icon,
                    factor_changed: variable.name,
                    original_value: String(currentValue),
                    suggested_value: String(nextValue)
                };
            }
        } catch (err) {
            console.error(`Error predicting scenario for ${variable.name}:`, err);
        }
    }

    return null;
}

async function runFrontendWhatIfAnalysis(formData, baselineHba1c) {
    const shapFactors = currentExplanation && currentExplanation.top_contributing_factors
        ? currentExplanation.top_contributing_factors
        : [];

    const eligibleVars = MODIFIABLE_VARS.filter(variable => {
        const val = formData[variable.name];
        if (val === undefined || val === null || val === '') return false;

        const gender = formData.PreBLGender;
        if (!variable.isUnfavourable(val, gender)) return false;

        const hasPositiveShap = shapFactors.some(f => {
            const baseFeature = f.feature.replace(/_\d+(\.\d+)?$/, '');
            return baseFeature === variable.name && f.impact > 0;
        });

        return hasPositiveShap;
    });

    const promises = eligibleVars.map(variable => simulateScenario(variable, formData, baselineHba1c));
    const results = await Promise.all(promises);
    return results.filter(r => r !== null);
}

function createFactorCard(factor, maxImpact, formData) {
    const card = document.createElement('div');
    card.className = 'factor-card';

    const isPositive = factor.impact > 0;
    const normalizedImpact = (Math.abs(factor.impact) / maxImpact) * 100;

    // Normalize OHE feature names to match base names
    const baseFeature = factor.feature.replace(/_\d+(\.\d+)?$/, '');

    // Append the mapped input value to the feature name
    let featureName = formatFeatureName(factor.feature);
    if (formData && formData[baseFeature] !== undefined) {
        const formattedValue = getFormattedInputValue(baseFeature, formData[baseFeature]);
        if (formattedValue) {
            featureName += ` (${formattedValue})`;
        }
    }

    // CHANGE 1 & 3: Override backend interpretation and apply Red Flag rules
    let interpretationText = factor.interpretation;
    const directionWord = isPositive ? 'higher' : 'lower';

    // Standardized overrides for all Table 1 and Table 2 features
    const overrideMap = {
        // Table 2 (Variables without dropdowns)
        'PostBLAge': `Based on your age, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreBLGender': `Based on your gender, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRarea': `Based on your place of residence, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRmaritalstatus': `Based on your marital status, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreReducation': `Based on your education level, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRpresentoccupation': `Based on your occupation, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRwaist': `Based on your waist circumference, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRBMI': `Based on your BMI, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRsystolicfirst': `Based on your systolic blood pressure, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRdiastolicfirst': `Based on your diastolic blood pressure, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreBLFBS': `Based on your fasting blood sugar, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreBLPPBS': `Based on your postprandial blood sugar, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreBLHBA1C': `Based on your current HbA1c level, the model predicts a slightly ${directionWord} HbA1c after the selected prediction period.`,
        'PreBLCHOLESTEROL': `Based on your total cholesterol level, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreBLTRIGLYCERIDES': `Based on your triglyceride level, the model predicts a slightly ${directionWord} HbA1c.`,
        'Diabetic_Duration': `Based on your duration of diabetes, the model predicts a slightly ${directionWord} HbA1c.`,
        'PostRgroupname': `Based on your selected care plan, the model predicts a slightly ${directionWord} HbA1c.`,

        // Table 1 (Variables with dropdowns / binary / categorical features)
        'PreRdiafather': `Based on your father's diabetes history, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRdiamother': `Based on your mother's diabetes history, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRdiabrother': `Based on your brother's diabetes history, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRdiasister': `Based on your sister's diabetes history, the model predicts a slightly ${directionWord} HbA1c.`,
        'current_smoking': `Based on your smoking status, the model predicts a slightly ${directionWord} HbA1c.`,
        'current_alcohol': `Based on your alcohol consumption, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRsleepquality': `Based on your sleep quality, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRmildactivityduration': `Based on your mild physical activity duration, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRmoderate': `Based on your moderate physical activity frequency, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRmoderateduration': `Based on your moderate physical activity duration, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRvigorous': `Based on your vigorous physical activity frequency, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRvigorousduration': `Based on your vigorous physical activity duration, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRskipbreakfast': `Based on your breakfast habits, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRlessfruit': `Based on your fruit consumption, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRlessvegetable': `Based on your vegetable consumption, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRmilk': `Based on your milk or curd intake, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRmeat': `Based on your meat or fish intake, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRfriedfood': `Based on your fried food intake, the model predicts a slightly ${directionWord} HbA1c.`,
        'PreRsweet': `Based on your sweet intake, the model predicts a slightly ${directionWord} HbA1c.`
    };

    if (overrideMap[baseFeature]) {
        interpretationText = overrideMap[baseFeature];
    }

    // Change 3: Red Flag Rules implementation using baseFeature and normalized checks
    const redFlagVars = ['PreRskipbreakfast', 'PreRlessfruit', 'PreRlessvegetable', 'PreRmilk', 'PreRfriedfood', 'PreRsweet'];
    let hasRedFlag = false; // NEW: Track if this card gets a red flag

    if (redFlagVars.includes(baseFeature)) {
        const val = formData ? formData[baseFeature] : null;
        const isLower = factor.impact < 0;
        const isHigher = factor.impact > 0;

        // Define the uniform warning text
        const warningText = "<br><br><span style='color: #E85D4C; font-weight: 600;'>🚩 Interpret with caution: This model result goes against usual clinical expectation. It does not mean this habit lowers HbA1c.</span>";

        if ((val === 1 && isLower) || (val === 2 && isLower) || (val === 3 && isHigher)) {
            interpretationText += warningText;
            hasRedFlag = true;
        }
    }

    card.dataset.hasRedFlag = hasRedFlag; // NEW: Store flag status in the DOM element

    card.innerHTML = `
        <div class="factor-header">
            <span class="factor-name">${featureName}</span>
            <span class="factor-direction ${isPositive ? 'increases' : 'reduces'}">
                ${factor.direction}
            </span>
        </div>
        <p class="factor-interpretation">${interpretationText}</p>
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
    const READABLE_NAMES = {
        "PostBLAge": "Age",
        "PreBLGender": "Gender",
        "PreRarea": "Residential area",
        "PreRmaritalstatus": "Marital status",
        "PreReducation": "Education level",
        "PreRpresentoccupation": "Current occupation",

        "PreRdiafather": "Father's diabetes history",
        "PreRdiamother": "Mother's diabetes history",
        "PreRdiabrother": "Brother's diabetes history",
        "PreRdiasister": "Sister's diabetes history",

        "PreRsleepquality": "Sleep quality",
        "PreRmildactivityduration": "Mild activity duration",
        "PreRmoderate": "Moderate physical activity",
        "PreRmoderateduration": "Moderate activity duration",
        "PreRvigorous": "Vigorous physical activity",
        "PreRvigorousduration": "Vigorous activity duration",

        "PreRskipbreakfast": "Skipping breakfast",
        "PreRlessfruit": "Low fruit intake",
        "PreRlessvegetable": "Low vegetable intake",
        "PreRmilk": "Milk consumption",
        "PreRmeat": "Meat consumption",
        "PreRfriedfood": "Fried food intake",
        "PreRsweet": "Sweet intake",

        "PreRwaist": "Waist circumference",
        "PreRBMI": "Body Mass Index (BMI)",

        "PreRsystolicfirst": "Systolic blood pressure",
        "PreRdiastolicfirst": "Diastolic blood pressure",

        "PreBLPPBS": "Postprandial blood glucose",
        "PreBLFBS": "Fasting blood glucose",
        "PreBLHBA1C": "HbA1c",

        "PreBLCHOLESTEROL": "Total cholesterol",
        "PreBLTRIGLYCERIDES": "Triglycerides",

        "Diabetic_Duration": "Duration of diabetes (years)",
        "PostRgroupname": "Intervention",
        "current_alcohol": "Current alcohol use",
        "current_smoking": "Current smoking status",
    };

    // Clean OHE suffixes from categorical variables (e.g. PreRmaritalstatus_1.0 -> PreRmaritalstatus)
    const baseName = name.split('_')[0] === name.split('_')[0] && !name.includes('group_x')
        ? name.replace(/_\d+(\.\d+)?$/, '')
        : name;

    if (READABLE_NAMES[baseName]) {
        // If it was a one-hot encoded variable, we might want to show the category value too,
        // but for SHAP the base name is often enough to represent "Impact of Marital Status".
        return READABLE_NAMES[baseName];
    }

    // Special handlers for interaction terms — rename to just "Intervention"
    if (name.includes('group_x_hba1c')) return 'Intervention';
    if (name.includes('group_x_fbs')) return 'Intervention';
    if (name.includes('group_x_ppbs')) return 'Intervention';
    // Hide any other group_x_ cross features
    if (name.includes('group_x_')) return 'Intervention';

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
    // Do NOT reset the form — persistence is handled via localStorage
    resetWhatIf();
    hideChatWidget();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// =============================================
// LOCAL STORAGE PERSISTENCE
// =============================================
const LS_KEY = 'diabsense_form_v1';

function saveFormToStorage(formData) {
    // Also capture radio + dropdown state from DOM directly
    const store = { ...formData };
    // Save radio button states by name
    ['PreBLGender', 'PreRarea', 'PreRdiafather', 'PreRdiamother',
        'PreRdiabrother', 'PreRdiasister', 'current_smoking', 'current_alcohol'
    ].forEach(name => {
        const checked = document.querySelector(`input[name="${name}"]:checked`);
        store['_radio_' + name] = checked ? checked.id : null;
    });
    try { localStorage.setItem(LS_KEY, JSON.stringify(store)); } catch (e) { }
}

function restoreFormFromStorage() {
    let store;
    try { store = JSON.parse(localStorage.getItem(LS_KEY)); } catch (e) { return; }
    if (!store) return;

    // Restore radio buttons
    ['PreBLGender', 'PreRarea', 'PreRdiafather', 'PreRdiamother',
        'PreRdiabrother', 'PreRdiasister', 'current_smoking', 'current_alcohol'
    ].forEach(name => {
        const id = store['_radio_' + name];
        if (id) {
            const el = document.getElementById(id);
            if (el) el.checked = true;
        }
    });

    // Restore direct fields
    const directFields = [
        'PostBLAge', 'PreRmaritalstatus', 'PreReducation', 'PreRpresentoccupation',
        'PreRsleepquality', 'PostRgroupname',
        'PreRmildactivity', 'PreRmildactivityduration', 'PreRmoderate', 'PreRmoderateduration',
        'PreRvigorous', 'PreRvigorousduration',
        'PreRskipbreakfast', 'PreRlessfruit', 'PreRlessvegetable',
        'PreRmilk', 'PreRmeat', 'PreRfriedfood', 'PreRsweet',
        'PreRwaist', 'PreRBMI', 'PreRsystolicfirst', 'PreRdiastolicfirst',
        'PreBLPPBS', 'PreBLFBS', 'PreBLHBA1C',
        'PreBLCHOLESTEROL', 'PreBLTRIGLYCERIDES', 'Diabetic_Duration',
    ];
    directFields.forEach(id => {
        const el = document.getElementById(id);
        if (el && store[id] !== undefined && store[id] !== null && !isNaN(store[id])) {
            el.value = store[id];
        }
    });
}

function resetForm() {
    try { localStorage.removeItem(LS_KEY); } catch (e) { }
    elements.patientForm.reset();
    showNotification('Form cleared.', 'info');
}

// =============================================
// INPUT SUMMARY PANEL
// =============================================

function toggleInputSummary() {
    const body = document.getElementById('inputSummaryBody');
    const chevron = document.getElementById('inputSummaryChevron');
    if (!body) return;
    const isOpen = body.style.display !== 'none';
    body.style.display = isOpen ? 'none' : 'block';
    if (chevron) chevron.style.transform = isOpen ? '' : 'rotate(180deg)';
}

function displayInputSummary(formData) {
    const grid = document.getElementById('inputSummaryGrid');
    if (!grid || !formData) return;

    const freqMap = { 0: 'None', 1: 'Once/month', 2: '2-3×/month', 3: 'Once/week', 4: '2-3×/week', 5: '4-5×/week', 6: 'Every day' };
    const durMap = { 0: 'None', 1: '≤10 min', 2: '10-30 min', 3: '30min-1hr', 4: '1-1.5hrs', 5: '>1.5hrs' };
    const dietMap = { 1: 'Usually/Often', 2: 'Sometimes', 3: 'Rarely/Never' };
    const maritalMap = { 1: 'Married', 2: 'Unmarried', 3: 'Divorcee/Separated', 4: 'Widow/Widower', 5: 'Others' };
    const eduMap = { 1: 'No schooling', 2: 'Primary', 3: 'High school', 4: 'Intermediate', 5: 'University', 6: 'Univ+', 7: 'Others' };
    const occMap = { 1: 'Professional', 2: 'Clerical', 3: 'Self-employed', 4: 'Unskilled', 5: 'Homemaker', 6: 'Retired', 7: 'Unemployed(able)', 8: 'Unemployed(unable)', 9: 'Others' };
    const sleepMap = { 1: 'Very good', 2: 'Fairly good', 3: 'Fairly bad', 4: 'Very bad' };
    const careMap = { 1: 'Standard + Yoga', 2: 'Standard care' };

    const radioLabel = (name) => document.querySelector(`input[name="${name}"]:checked`)?.parentElement?.textContent?.trim() || '—';

    const rows = [
        ['Age', formData.PostBLAge + ' yrs'],
        ['Gender', formData.PreBLGender || radioLabel('PreBLGender')],
        ['Residence', formData.PreRarea === 1 ? 'Urban' : 'Rural'],
        ['Marital Status', maritalMap[formData.PreRmaritalstatus] || '—'],
        ['Education', eduMap[formData.PreReducation] || '—'],
        ['Occupation', occMap[formData.PreRpresentoccupation] || '—'],
        ['Diabetic Father', formData.PreRdiafather ? 'Yes' : 'No'],
        ['Diabetic Mother', formData.PreRdiamother ? 'Yes' : 'No'],
        ['Diabetic Brother', formData.PreRdiabrother ? 'Yes' : 'No'],
        ['Diabetic Sister', formData.PreRdiasister ? 'Yes' : 'No'],
        ['Smoking', formData.current_smoking ? 'Yes' : 'No'],
        ['Alcohol', formData.current_alcohol ? 'Yes' : 'No'],
        ['Sleep Quality', sleepMap[formData.PreRsleepquality] || '—'],
        ['Care Plan', careMap[formData.PostRgroupname] || '—'],
        ['Mild Activity (freq)', freqMap[formData.PreRmildactivity] || freqMap[document.getElementById('PreRmildactivity')?.value] || '—'],
        ['Mild Activity (dur)', durMap[formData.PreRmildactivityduration] || '—'],
        ['Moderate Activity (freq)', freqMap[formData.PreRmoderate] || '—'],
        ['Moderate Activity (dur)', durMap[formData.PreRmoderateduration] || '—'],
        ['Vigorous Activity (freq)', freqMap[formData.PreRvigorous] || '—'],
        ['Vigorous Activity (dur)', durMap[formData.PreRvigorousduration] || '—'],
        ['Skip Breakfast', dietMap[formData.PreRskipbreakfast] || '—'],
        ['Less Fruit', dietMap[formData.PreRlessfruit] || '—'],
        ['Less Vegetable', dietMap[formData.PreRlessvegetable] || '—'],
        ['Milk/Curd <400ml', dietMap[formData.PreRmilk] || '—'],
        ['Meat/Fish >250g', dietMap[formData.PreRmeat] || '—'],
        ['Fried Food', dietMap[formData.PreRfriedfood] || '—'],
        ['Sweets >2×/day', dietMap[formData.PreRsweet] || '—'],
        ['Waist', formData.PreRwaist + ' cm'],
        ['BMI', formData.PreRBMI + ' kg/m²'],
        ['SBP', formData.PreRsystolicfirst + ' mmHg'],
        ['DBP', formData.PreRdiastolicfirst + ' mmHg'],
        ['PPBS', formData.PreBLPPBS + ' mg/dL'],
        ['FBS', formData.PreBLFBS + ' mg/dL'],
        ['HbA1c (baseline)', formData.PreBLHBA1C + '%'],
        ['Cholesterol', formData.PreBLCHOLESTEROL + ' mg/dL'],
        ['Triglycerides', formData.PreBLTRIGLYCERIDES + ' mg/dL'],
        ['Diabetic Duration', formData.Diabetic_Duration + ' yrs'],
    ];

    grid.innerHTML = rows.map(([label, val]) => `
        <div style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:0.65rem;color:#888;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:2px;">${label}</div>
            <div style="font-size:0.85rem;color:#f5f0e8;font-weight:500;">${val}</div>
        </div>
    `).join('');
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
    // Fake mild activity frequency (not sent to backend)
    const mildActEl = document.getElementById('PreRmildactivity');
    if (mildActEl) mildActEl.value = '2';
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
        const baselineHba1c = whatifData.original_hba1c;

        // Keep backend intervention scenarios (PostRgroupname or group_x_)
        const backendInterventions = (whatifData.scenarios || []).filter(s => {
            const feat = s.feature || s.factor_changed || '';
            return feat === 'PostRgroupname' || feat.includes('group_x_');
        });

        // Generate other modifiable scenarios on the frontend
        const frontendScenarios = await runFrontendWhatIfAnalysis(formData, baselineHba1c);

        // Combine both
        const allScenarios = [...backendInterventions, ...frontendScenarios];

        // Recalculate combined outcome if we have multiple scenarios
        let combinedHba1c = null;
        let combinedRiskLevel = null;

        if (allScenarios.length > 1) {
            const combinedData = { ...formData };
            allScenarios.forEach(s => {
                const val = s.suggested_value;
                combinedData[s.factor_changed] = isNaN(val) ? val : Number(val);
            });

            try {
                const combinedResponse = await fetch(`${API_BASE_URL}/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(combinedData)
                });
                if (combinedResponse.ok) {
                    const combinedResult = await combinedResponse.json();
                    combinedHba1c = combinedResult.predicted_hba1c;
                    combinedRiskLevel = getRiskLevelString(combinedHba1c);
                }
            } catch (err) {
                console.error("Error calculating combined outcome:", err);
            }
        } else if (allScenarios.length === 1) {
            combinedHba1c = allScenarios[0].modified_hba1c;
            combinedRiskLevel = getRiskLevelString(combinedHba1c);
        }

        // Sort: Care Plan / Intervention first, then by reduction descending
        allScenarios.sort((a, b) => {
            const aFeat = a.feature || a.factor_changed || '';
            const bFeat = b.feature || b.factor_changed || '';
            const aIsIntervention = aFeat === 'PostRgroupname' || aFeat.includes('group_x_');
            const bIsIntervention = bFeat === 'PostRgroupname' || bFeat.includes('group_x_');
            if (aIsIntervention && !bIsIntervention) return -1;
            if (bIsIntervention && !aIsIntervention) return 1;
            return b.hba1c_delta - a.hba1c_delta;
        });

        allScenarios.forEach((s, idx) => {
            s.scenario_id = idx + 1;
        });

        const mergedWhatifData = {
            original_hba1c: baselineHba1c,
            original_risk_level: whatifData.original_risk_level,
            scenarios: allScenarios,
            best_scenario: allScenarios.length > 0 ? allScenarios[0] : null,
            combined_hba1c: combinedHba1c,
            combined_risk_level: combinedRiskLevel
        };

        currentWhatIf = mergedWhatifData;
        elements.whatifLoading.style.display = 'none';
        displayWhatIfAnalysis(mergedWhatifData);
    } catch (error) {
        console.error('What-If API Error:', error);
        elements.whatifLoading.style.display = 'none';
        elements.whatifScenariosGrid.innerHTML = `
            <div class="whatif-error"><p>⚠️ Could not generate What-If scenarios.</p></div>
        `;
    }
}

function displayWhatIfAnalysis(data) {
    const validScenarios = data.scenarios || [];

    if (!validScenarios || validScenarios.length === 0) {
        const isHealthy = data.original_hba1c && data.original_hba1c < 5.7;
        const msg = isHealthy
            ? "✅ Your current health parameters are already in healthy ranges!"
            : "⚠️ No single lifestyle change was predicted to significantly lower your HbA1c further on its own. Consider combining multiple interventions.";
        elements.whatifScenariosGrid.innerHTML = `
            <div class="whatif-empty"><p>${msg}</p></div>
        `;
        return;
    }

    validScenarios.forEach((scenario, index) => {
        const card = createScenarioCard(scenario);
        elements.whatifScenariosGrid.appendChild(card);
        setTimeout(() => { card.classList.add('animate'); }, 100 + index * 150);
    });

    if (data.combined_hba1c !== null && data.combined_hba1c !== undefined && validScenarios.length > 1) {
        setTimeout(() => {
            displayCombinedOutcome(data);
        }, 100 + validScenarios.length * 150 + 200);
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

    // Setup care plan dropdown
    const carePlanSelect = document.getElementById('simCarePlan');
    if (carePlanSelect && currentFormData) {
        carePlanSelect.value = String(currentFormData.PostRgroupname || 1);
        carePlanSelect.onchange = () => { simulateRisk(); };
    }

    // Setup Pre-HbA1c slider
    try {
        const slider = document.getElementById('simHba1cSlider');
        const num = document.getElementById('simHba1cNum');
        if (slider && num && currentFormData) {
            let val = Number(currentFormData.PreBLHBA1C);
            if (isNaN(val) || val < 4) val = 4;
            if (val > 18) val = 18;
            slider.value = val;
            num.value = val;
            num.setAttribute('min', 4);
            num.setAttribute('max', 18);
            num.setAttribute('step', '0.1');
            slider.oninput = (e) => { num.value = e.target.value; triggerSimulate(); };
            num.oninput = (e) => {
                let v = Number(e.target.value);
                if (!isNaN(v) && v >= 4 && v <= 18) { slider.value = v; triggerSimulate(); }
            };
        }
    } catch (err) {
        console.error('[Sim] Failed to setup Pre-HbA1c:', err);
    }
}

function triggerSimulate() {
    clearTimeout(simTimeout);
    simTimeout = setTimeout(simulateRisk, 300);
}

async function simulateRisk() {
    if (!currentFormData) return;

    const simulatedData = { ...currentFormData };

    // Change care plan (intervention)
    const carePlanSelect = document.getElementById('simCarePlan');
    if (carePlanSelect) {
        simulatedData.PostRgroupname = parseInt(carePlanSelect.value) || currentFormData.PostRgroupname;
    }

    // Change Pre-HbA1c
    const hba1cNum = document.getElementById('simHba1cNum');
    if (hba1cNum) {
        const v = parseFloat(hba1cNum.value);
        if (!isNaN(v)) simulatedData.PreBLHBA1C = v;
    }

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
        addChatMessage('system', '⚠️ Could not connect to DiabSense AI. Make sure NVIDIA_API_KEY is set and agno is installed.');
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
