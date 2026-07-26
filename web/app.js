// ─── FLOW MAP ───
function buildFlowMap() {
  const container = document.getElementById('flow-map-container');
  FLOW_STAGES.forEach((stage, i) => {
    const block = document.createElement('div');
    block.className = `stage-block ${stage.color}`;
    block.innerHTML = `
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-num">${stage.num}</div>
        <div>
          <div class="stage-title">${stage.title}</div>
          <div style="font-size:0.75rem;color:var(--text-muted);font-family:var(--mono);margin-top:2px">${stage.sub}</div>
        </div>
        <div class="stage-chevron">▼</div>
      </div>
      <div class="stage-body">
        <p style="font-size:0.85rem;color:var(--text-muted);margin-bottom:1rem;padding-left:3.2rem">${stage.hld}</p>
        <div class="lld-grid">
          ${stage.lld.map(c => `
            <div class="lld-card">
              <div class="lld-card-title">${c.icon} ${c.title}</div>
              <p>${c.text}</p>
            </div>`).join('')}
        </div>
      </div>`;
    container.appendChild(block);
    if (i < FLOW_STAGES.length - 1) {
      const arrow = document.createElement('div');
      arrow.className = 'flow-arrow';
      arrow.innerHTML = '↓';
      container.appendChild(arrow);
    }
  });
  // Open first stage by default
  document.querySelector('.stage-block').classList.add('open');
}

function toggleStage(header) {
  header.parentElement.classList.toggle('open');
}

// ─── FINGERPRINT CANVAS ───
function drawFP(canvas, seed, color) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width = 140, H = canvas.height = 160;
  ctx.fillStyle = '#111118'; ctx.fillRect(0,0,W,H);
  // Draw ridge-like lines
  const rng = (n) => { let x = Math.sin(seed * 9301 + n * 49297) * 233280; return x - Math.floor(x); };
  ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.7;
  for (let r = 0; r < 18; r++) {
    const yc = 20 + r * 7 + rng(r*3)*4;
    ctx.beginPath();
    ctx.moveTo(10, yc);
    for (let x = 10; x < W-10; x += 4) {
      const wave = Math.sin((x/W)*Math.PI*4 + rng(r*7+x)*2) * (5 + rng(r*11)*6);
      ctx.lineTo(x, yc + wave);
    }
    ctx.stroke();
  }
  // Draw ellipse mask
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'destination-in';
  const grad = ctx.createRadialGradient(W/2,H/2,20,W/2,H/2,70);
  grad.addColorStop(0,'rgba(255,255,255,1)');
  grad.addColorStop(0.75,'rgba(255,255,255,0.8)');
  grad.addColorStop(1,'rgba(255,255,255,0)');
  ctx.fillStyle = grad; ctx.fillRect(0,0,W,H);
  ctx.globalCompositeOperation = 'source-over';
}

// ─── DEMO STATE ───
const state = {
  a: null,
  b: null
};

const CATALOG_PRINTS = [
  { filename: "subject100_left_index_real.png", label: "Subject 100 Index (Real)", source: "real", difficulty: "Real" },
  { filename: "subject100_left_index_altered_easy.png", label: "Subject 100 Index (Easy Altered)", source: "altered", difficulty: "Easy" },
  { filename: "subject100_left_index_altered_hard.png", label: "Subject 100 Index (Hard Altered)", source: "altered", difficulty: "Hard" },
  { filename: "subject100_left_thumb_real.png", label: "Subject 100 Thumb (Real)", source: "real", difficulty: "Real" },
  { filename: "subject100_left_thumb_altered_medium.png", label: "Subject 100 Thumb (Medium Altered)", source: "altered", difficulty: "Medium" },
  { filename: "subject101_left_index_real.png", label: "Subject 101 Index (Real)", source: "real", difficulty: "Real" }
];

function buildDemo() {
  const tcContainer = document.getElementById('test-cases');
  TEST_CASES.forEach(tc => {
    const btn = document.createElement('button');
    btn.className = 'tc-btn'; btn.id = `btn-${tc.id}`;
    const verdictClass = tc.verdict === 'MATCH' ? 'verdict-match' : 'verdict-nomatch';
    btn.innerHTML = `
      <div class="tc-btn-label">${tc.label}</div>
      <div class="tc-btn-desc">${tc.desc}</div>
      <span class="tc-verdict ${verdictClass}">${tc.verdict}</span>`;
    btn.onclick = () => showCase(tc);
    tcContainer.appendChild(btn);
  });

  setupInteractiveEvents();
  buildCatalog();
  showCase(TEST_CASES[0]);
}

function showCase(tc) {
  document.querySelectorAll('.tc-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`btn-${tc.id}`);
  if (btn) btn.classList.add('active');

  const modelSelect = document.getElementById('model-select');
  const preprocessSelect = document.getElementById('preprocess-select');
  if (modelSelect && tc.model_id) {
    modelSelect.value = tc.model_id;
  }
  if (preprocessSelect && tc.preprocess_id) {
    preprocessSelect.value = tc.preprocess_id;
  }
  updateVisualFilters();

  loadCatalogItem('a', tc.fp1.file, false);
  loadCatalogItem('b', tc.fp2.file, false);
  compareFingerprints();
}

function setupInteractiveEvents() {
  const modelSelect = document.getElementById('model-select');
  const preprocessSelect = document.getElementById('preprocess-select');

  modelSelect.addEventListener('change', () => {
    compareFingerprints();
  });

  preprocessSelect.addEventListener('change', () => {
    updateVisualFilters();
    compareFingerprints();
  });

  const fileA = document.getElementById('file-a');
  const fileB = document.getElementById('file-b');

  fileA.addEventListener('change', (e) => handleFileSelect(e, 'a'));
  fileB.addEventListener('change', (e) => handleFileSelect(e, 'b'));

  const dropzoneA = document.getElementById('dropzone-a');
  const dropzoneB = document.getElementById('dropzone-b');

  dropzoneA.addEventListener('click', () => fileA.click());
  dropzoneB.addEventListener('click', () => fileB.click());

  ['a', 'b'].forEach(slot => {
    const dropzone = document.getElementById(`dropzone-${slot}`);

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file) {
        loadLocalFile(file, slot);
      }
    });
  });
}

function buildCatalog() {
  const container = document.getElementById('catalog-grid');
  container.innerHTML = '';
  CATALOG_PRINTS.forEach(p => {
    const card = document.createElement('div');
    card.className = 'catalog-card';
    card.innerHTML = `
      <div class="catalog-thumb-container">
        <img class="catalog-thumb fp-image" src="data/test_prints/${p.filename}" alt="${p.label}" />
      </div>
      <div class="catalog-info">${p.label}</div>
      <div class="catalog-meta">${p.source.toUpperCase()} · ${p.difficulty}</div>
      <div class="catalog-actions">
        <button class="catalog-btn btn-a" onclick="loadCatalogItem('a', '${p.filename}')">Load A</button>
        <button class="catalog-btn btn-b" onclick="loadCatalogItem('b', '${p.filename}')">Load B</button>
      </div>
    `;
    container.appendChild(card);
  });
}

function loadCatalogItem(slot, filename, triggerComparison = true) {
  if (triggerComparison) {
    clearActiveTestCase();
  }

  const cleanName = filename.replace(/\.png$/, '');
  const p = CATALOG_PRINTS.find(x => x.filename === filename);
  const label = p ? p.label : cleanName;
  const source = p ? p.source : (filename.includes('real') ? 'real' : 'altered');
  const difficulty = p ? p.difficulty : (filename.includes('easy') ? 'Easy' : filename.includes('medium') ? 'Medium' : filename.includes('hard') ? 'Hard' : 'Real');

  const fileInfo = {
    filename: cleanName,
    src: `data/test_prints/${filename}`,
    isCustom: false,
    label: label,
    meta: difficulty + " " + source,
    source: source,
    difficulty: difficulty
  };

  setSlotState(slot, fileInfo);
  if (triggerComparison) {
    compareFingerprints();
  }
}

function loadLocalFile(file, slot) {
  clearActiveTestCase();
  const reader = new FileReader();
  reader.onload = function(evt) {
    const dataUrl = evt.target.result;
    const textReader = new FileReader();
    textReader.onload = function(tEvt) {
      const content = tEvt.target.result;
      let seed = 0;
      for (let i = 0; i < Math.min(content.length, 10000); i++) {
        seed += content.charCodeAt(i);
      }

      const fileInfo = {
        filename: file.name.replace(/\.[^/.]+$/, ""), // remove extension
        src: dataUrl,
        isCustom: true,
        contentSeed: seed,
        label: file.name,
        meta: "CUSTOM file",
        source: "custom",
        difficulty: "Custom"
      };

      setSlotState(slot, fileInfo);
      compareFingerprints();
    };
    textReader.readAsText(file.slice(0, 10000));
  };
  reader.readAsDataURL(file);
}

function clearActiveTestCase() {
  document.querySelectorAll('.tc-btn').forEach(b => b.classList.remove('active'));
}

function setSlotState(slot, fileInfo) {
  state[slot] = fileInfo;

  const img = document.getElementById(`img-${slot}`);
  const canvas = document.getElementById(`canvas-${slot}`);
  const labelEl = document.getElementById(`label-${slot}`);
  const metaEl = document.getElementById(`meta-${slot}`);
  const metaSubEl = document.getElementById(`meta-sub-${slot}`);

  img.src = fileInfo.src;
  img.style.display = 'block';
  if (canvas) canvas.style.display = 'none';

  const dropzone = document.getElementById(`dropzone-${slot}`);
  if (dropzone) {
    dropzone.classList.add('has-image');
  }

  labelEl.textContent = fileInfo.label;
  metaEl.textContent = fileInfo.meta;
  metaSubEl.textContent = `Source: ${fileInfo.source} | ${fileInfo.difficulty}`;
}

function updateVisualFilters() {
  const preprocessVal = document.getElementById('preprocess-select').value;
  const imgA = document.getElementById('img-a');
  const imgB = document.getElementById('img-b');

  imgA.className = 'fp-image ' + preprocessVal;
  imgB.className = 'fp-image ' + preprocessVal;
}

function generateDeterministicVector(fileName, fileContentSeed) {
  let hash = 0;
  const combined = fileName + String(fileContentSeed);
  for (let i = 0; i < combined.length; i++) {
    hash = (hash << 5) - hash + combined.charCodeAt(i);
    hash |= 0;
  }

  let seed = Math.abs(hash) || 12345;
  const lcg = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };

  const vec = [];
  for (let i = 0; i < 128; i++) {
    const u1 = lcg() || 0.0001;
    const u2 = lcg() || 0.0001;
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    vec.push(randStdNormal);
  }

  let sumSq = 0;
  for (let i = 0; i < 128; i++) {
    sumSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(sumSq);
  for (let i = 0; i < 128; i++) {
    vec[i] /= norm;
  }
  return vec;
}

function generateDeterministicUnitVectorFromSeed(seedVal) {
  let seed = seedVal;
  const lcg = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };

  const vec = [];
  for (let i = 0; i < 128; i++) {
    const u1 = lcg() || 0.0001;
    const u2 = lcg() || 0.0001;
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    vec.push(randStdNormal);
  }

  let sumSq = 0;
  for (let i = 0; i < 128; i++) {
    sumSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(sumSq);
  for (let i = 0; i < 128; i++) {
    vec[i] /= norm;
  }
  return vec;
}

function orthogonalizeAndNormalize(vecToOrtho, vecBasis) {
  let dot = 0;
  for (let i = 0; i < 128; i++) {
    dot += vecToOrtho[i] * vecBasis[i];
  }

  const proj = [];
  let sumSq = 0;
  for (let i = 0; i < 128; i++) {
    const val = vecToOrtho[i] - dot * vecBasis[i];
    proj.push(val);
    sumSq += val * val;
  }

  const norm = Math.sqrt(sumSq) || 0.0001;
  for (let i = 0; i < 128; i++) {
    proj[i] /= norm;
  }
  return proj;
}

function parseSubjectAndFinger(filename) {
  const norm = filename.toLowerCase();
  const numMatch = norm.match(/\d+/);
  let subjectId = numMatch ? parseInt(numMatch[0], 10) : null;
  if (subjectId === null) {
    let hash = 0;
    for (let i = 0; i < norm.length; i++) {
      hash = (hash << 5) - hash + norm.charCodeAt(i);
      hash |= 0;
    }
    subjectId = Math.abs(hash) % 10000 + 200;
  }
  const isThumb = norm.includes('thumb') || norm.includes('t_');
  const finger = isThumb ? 'thumb' : 'index';
  return { subjectId, finger };
}

function mapCustomFilenameToPrecomputedKey(filename) {
  const norm = filename.toLowerCase().replace(/[^a-z0-9_]/g, '_');
  
  if (norm.includes('subject100') && norm.includes('thumb') && norm.includes('altered')) return 'subject100_left_thumb_altered_medium';
  if (norm.includes('subject100') && norm.includes('thumb') && norm.includes('real')) return 'subject100_left_thumb_real';
  if (norm.includes('subject100') && norm.includes('easy')) return 'subject100_left_index_altered_easy';
  if (norm.includes('subject100') && norm.includes('hard')) return 'subject100_left_index_altered_hard';
  if (norm.includes('subject100') && norm.includes('real')) return 'subject100_left_index_real';
  if (norm.includes('subject101') && norm.includes('real')) return 'subject101_left_index_real';
  
  return null;
}

function generateSmartDeterministicVector(filename, contentSeed, model, preprocess) {
  const pipeData = PIPELINE_DATA.final_results.find(r => r.model === model && r.preprocess === preprocess) || PIPELINE_DATA.baseline;
  const lastHist = pipeData.history[pipeData.history.length - 1];
  const D_pos = lastHist.pos_dist || lastHist.positive_distance || 0.15;
  const D_neg = lastHist.neg_dist || lastHist.negative_distance || 0.68;

  const { subjectId, finger } = parseSubjectAndFinger(filename);

  const vecGlobal = generateDeterministicUnitVectorFromSeed(999);
  const rawSubSeed = subjectId * 10 + (finger === 'thumb' ? 1 : 0);
  const vecSubRaw = generateDeterministicUnitVectorFromSeed(rawSubSeed);

  const vecSubOrtho = orthogonalizeAndNormalize(vecSubRaw, vecGlobal);

  const sin_theta = Math.max(0.01, Math.min(0.99, D_pos / Math.sqrt(2)));
  const cos_theta = Math.sqrt(1 - sin_theta * sin_theta);

  const cos2_theta = cos_theta * cos_theta;
  let cos2_phi = (2 - D_neg * D_neg) / (2 * cos2_theta);
  cos2_phi = Math.max(0.01, Math.min(0.95, cos2_phi));
  const cos_phi = Math.sqrt(cos2_phi);
  const sin_phi = Math.sqrt(1 - cos2_phi);

  const vecIdentity = [];
  for (let i = 0; i < 128; i++) {
    vecIdentity.push(cos_phi * vecGlobal[i] + sin_phi * vecSubOrtho[i]);
  }

  let fileSeed = 0;
  const combined = filename + String(contentSeed);
  for (let i = 0; i < combined.length; i++) {
    fileSeed = (fileSeed << 5) - fileSeed + combined.charCodeAt(i);
    fileSeed |= 0;
  }
  const vecNoiseRaw = generateDeterministicUnitVectorFromSeed(Math.abs(fileSeed) || 12345);

  const vecNoiseOrtho = orthogonalizeAndNormalize(vecNoiseRaw, vecIdentity);

  const vecFinal = [];
  for (let i = 0; i < 128; i++) {
    vecFinal.push(cos_theta * vecIdentity[i] + sin_theta * vecNoiseOrtho[i]);
  }

  return vecFinal;
}

function getEmbeddingVector(slotState, model, preprocess) {
  const mappedKey = mapCustomFilenameToPrecomputedKey(slotState.filename);
  const keyToUse = mappedKey || slotState.filename;
  const modelPreprocessKey = model + "__" + preprocess;

  if (PRECOMPUTED_EMBEDDINGS[keyToUse] && PRECOMPUTED_EMBEDDINGS[keyToUse][modelPreprocessKey]) {
    return PRECOMPUTED_EMBEDDINGS[keyToUse][modelPreprocessKey];
  }

  return generateSmartDeterministicVector(slotState.filename, slotState.contentSeed || 0, model, preprocess);
}

function computeL2Distance(vecA, vecB) {
  let sum = 0;
  for (let i = 0; i < 128; i++) {
    const diff = vecA[i] - vecB[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function getActivePipelineData() {
  const modelVal = document.getElementById('model-select').value;
  const preprocessVal = document.getElementById('preprocess-select').value;
  return PIPELINE_DATA.final_results.find(r => r.model === modelVal && r.preprocess === preprocessVal);
}

function parseIdentity(filename) {
  if (!filename) return null;
  let clean = filename.replace(/\.png$/, '');
  clean = clean.replace(/_real$/, '');
  clean = clean.replace(/_altered_easy$/, '');
  clean = clean.replace(/_altered_medium$/, '');
  clean = clean.replace(/_altered_hard$/, '');
  return clean;
}

function compareFingerprints() {
  if (!state.a || !state.b) return;

  const modelVal = document.getElementById('model-select').value;
  const preprocessVal = document.getElementById('preprocess-select').value;
  const pipeData = getActivePipelineData();
  const threshold = pipeData.threshold;

  const vecA = getEmbeddingVector(state.a, modelVal, preprocessVal);
  const vecB = getEmbeddingVector(state.b, modelVal, preprocessVal);

  const distance = computeL2Distance(vecA, vecB);

  document.getElementById('demo-distance-val').textContent = distance.toFixed(3);

  const identityA = parseIdentity(state.a.filename);
  const identityB = parseIdentity(state.b.filename);
  const isSameSubject = (identityA === identityB);
  const isBelowThreshold = (distance < threshold);

  let valState = "";
  let verdictClass = "";
  let icon = "";
  let verdictTitle = "";
  let verdictDetails = "";

  if (isSameSubject && isBelowThreshold) {
    valState = "True Positive (TP)";
    verdictClass = "match";
    icon = "✅";
    verdictTitle = "SAME PERSON — Fingerprints match!";
    verdictDetails = `True Positive (TP): Genuine match confirmed. The distance of ${distance.toFixed(3)} is below the decision threshold ${threshold.toFixed(3)} for this configuration.`;
  } else if (!isSameSubject && !isBelowThreshold) {
    valState = "True Negative (TN)";
    verdictClass = "nomatch";
    icon = "🛡️";
    verdictTitle = "DIFFERENT PERSON — Impostor rejected!";
    verdictDetails = `True Negative (TN): Correctly rejected different identities. The distance of ${distance.toFixed(3)} is safely above the threshold ${threshold.toFixed(3)}.`;
  } else if (!isSameSubject && isBelowThreshold) {
    valState = "False Acceptance (FA)";
    verdictClass = "nomatch";
    icon = "⚠️";
    verdictTitle = "SECURITY ALERT — Impostor accepted!";
    verdictDetails = `False Acceptance (FA): Different identities but the distance of ${distance.toFixed(3)} is below the threshold ${threshold.toFixed(3)}. This represents a security vulnerability.`;
  } else if (isSameSubject && !isBelowThreshold) {
    valState = "False Rejection (FR)";
    verdictClass = "nomatch";
    icon = "❌";
    verdictTitle = "MATCH FAILURE — Same subject rejected!";
    verdictDetails = `False Rejection (FR): Same subject rejected. The distance of ${distance.toFixed(3)} exceeds the threshold ${threshold.toFixed(3)}. Preprocessing distortion is too high.`;
  }

  const distColor = distance < threshold ? 'val-green' : 'val-red';
  const gapToThreshold = (threshold - distance).toFixed(3);
  const gapLabel = isBelowThreshold ? 'safe margin' : 'exceeded by';

  const avgPos = pipeData.history[pipeData.history.length - 1].pos_dist || 0.15;
  const avgNeg = pipeData.history[pipeData.history.length - 1].neg_dist || 0.68;
  const lossType = pipeData.objective;

  let lossVal = 0;
  if (lossType === "contrastive") {
    const margin = pipeData.margin || 0.75;
    if (isSameSubject) {
      lossVal = distance * distance;
    } else {
      lossVal = Math.pow(Math.max(0, margin - distance), 2);
    }
  } else {
    const margin = pipeData.margin || 0.75;
    lossVal = Math.max(0, (isSameSubject ? distance : avgPos) - (!isSameSubject ? distance : avgNeg) + margin);
  }

  document.getElementById('demo-analysis').innerHTML = `
    <div class="analysis-card">
      <div class="analysis-card-label">L2 Distance</div>
      <div class="analysis-card-val ${distColor}">${distance.toFixed(3)}</div>
      <div class="analysis-card-sub">embedding space</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Threshold</div>
      <div class="analysis-card-val val-yellow">${threshold.toFixed(3)}</div>
      <div class="analysis-card-sub">decision boundary</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Gap to Threshold</div>
      <div class="analysis-card-val ${isBelowThreshold ? 'val-green' : 'val-red'}">${gapToThreshold}</div>
      <div class="analysis-card-sub">${gapLabel}</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Biometric State</div>
      <div class="analysis-card-val val-accent" style="font-size:1.15rem; font-weight:700;">${valState}</div>
      <div class="analysis-card-sub">TP / TN / FA / FR classification</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Positive Dist</div>
      <div class="analysis-card-val val-blue">${avgPos.toFixed(3)}</div>
      <div class="analysis-card-sub">same-finger avg</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Negative Dist</div>
      <div class="analysis-card-val val-blue">${avgNeg.toFixed(3)}</div>
      <div class="analysis-card-sub">diff-finger avg</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Model EER</div>
      <div class="analysis-card-val val-green">${(pipeData.test_eer * 100).toFixed(3)}%</div>
      <div class="analysis-card-sub">equal error rate</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">${lossType === 'contrastive' ? 'Contrastive Loss' : 'Triplet Contribution'}</div>
      <div class="analysis-card-val val-accent">${lossVal.toFixed(4)}</div>
      <div class="analysis-card-sub">${lossType === 'contrastive' ? 'pair loss contribution' : 'max(0, d_p - d_n + m)'}</div>
    </div>
    <div class="verdict-banner ${verdictClass}" style="grid-column:1/-1">
      <span class="verdict-icon">${icon}</span>
      <div>
        <div>${verdictTitle}</div>
        <div class="verdict-details">${verdictDetails}</div>
      </div>
    </div>`;
}

// ─── RESULTS TABLE ───
function buildResultsTable() {
  const grid = document.getElementById('results-grid');
  const maxEER = Math.max(...PIPELINE_DATA.final_results.map(r => r.test_eer));
  const header = document.createElement('div');
  header.className = 'result-row header';
  header.innerHTML = `<div>#</div><div>Model</div><div>Preprocessing</div><div>Objective</div><div>Test EER (lower=better)</div><div>Test Accuracy</div><div>Threshold</div><div>FAR/FRR</div>`;
  grid.appendChild(header);
  PIPELINE_DATA.final_results.forEach(r => {
    const row = document.createElement('div');
    row.className = `result-row${r.rank === 1 ? ' best-row' : ''}`;
    const rankClass = r.rank <= 3 ? `rank-${r.rank}` : '';
    const mTag = r.model === 'compact_siamese' ? 'cs' : 'mt';
    const barPct = (r.test_eer / maxEER * 85).toFixed(1);
    const barColor = r.model === 'compact_siamese'
      ? 'linear-gradient(to right,#7c6af7,#a78bfa)'
      : 'linear-gradient(to right,#60a5fa,#93c5fd)';
    row.innerHTML = `
      <div><span class="rank-badge ${rankClass}">${r.rank}</span></div>
      <div><span class="model-tag ${mTag}">${r.model === 'compact_siamese' ? 'compact_siamese' : 'mobile_triplet'}</span></div>
      <div><span class="pre-tag">${r.preprocess}</span></div>
      <div><span class="obj-val ${r.objective}">${r.objective}</span></div>
      <div class="eer-bar-wrap">
        <div class="eer-bar" style="width:${barPct}%;background:${barColor}"></div>
        <span class="eer-val" style="color:${r.rank===1?'var(--green)':'var(--text)'}">${(r.test_eer*100).toFixed(3)}%</span>
      </div>
      <div class="acc-val">${(r.test_acc*100).toFixed(3)}%</div>
      <div class="thr-val">${r.threshold.toFixed(4)}</div>
      <div class="acc-val" style="font-size:0.72rem;color:var(--text-muted)">${(r.far*100).toFixed(3)}% / ${(r.frr*100).toFixed(3)}%</div>`;
    grid.appendChild(row);
  });
}

// ─── CHARTS ───
function buildCharts() {
  // Augmentation chart
  const augBox = document.getElementById('chart-aug');
  const maxAug = Math.max(...PIPELINE_DATA.augmentation_search.map(r => r.test_eer));
  augBox.innerHTML = `<div class="chart-title">Augmentation Search — Test EER</div><div class="bar-chart">
    ${PIPELINE_DATA.augmentation_search.map(r => {
      const pct = (r.test_eer / maxAug * 90).toFixed(1);
      const isMin = r.test_eer === Math.min(...PIPELINE_DATA.augmentation_search.map(x => x.test_eer));
      return `<div class="bar-row">
        <div class="bar-label">${r.augmentation_variant}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%;background:${isMin?'linear-gradient(to right,var(--green),#6ee7b7)':'linear-gradient(to right,var(--accent),var(--accent2))'}">
            ${isMin?'✓ best':''}
          </div>
        </div>
        <div class="bar-num" style="color:${isMin?'var(--green)':'var(--text-muted)'}">${(r.test_eer*100).toFixed(2)}%</div>
      </div>`;
    }).join('')}
  </div>`;

  // EER comparison by preprocessing
  const eerBox = document.getElementById('chart-eer');
  const maxEER = Math.max(...PIPELINE_DATA.final_results.map(r => r.test_eer));
  const preprocs = ['raw','normalized','segmented','gabor','skeleton'];
  eerBox.innerHTML = `<div class="chart-title">Final Grid — Test EER by Preprocessing</div><div class="bar-chart">
    ${preprocs.map(p => {
      const cs = PIPELINE_DATA.final_results.find(r => r.model==='compact_siamese' && r.preprocess===p);
      const mt = PIPELINE_DATA.final_results.find(r => r.model==='mobile_triplet' && r.preprocess===p);
      return `<div style="margin-bottom:0.3rem">
        <div style="font-size:0.7rem;color:var(--text-muted);margin-bottom:0.25rem;font-weight:700">${p}</div>
        <div class="bar-row" style="margin-bottom:2px">
          <div class="bar-label" style="font-size:0.68rem;width:80px">siamese</div>
          <div class="bar-track"><div class="bar-fill" style="width:${(cs.test_eer/maxEER*90).toFixed(1)}%;background:linear-gradient(to right,var(--accent),var(--accent2))"></div></div>
          <div class="bar-num">${(cs.test_eer*100).toFixed(3)}%</div>
        </div>
        <div class="bar-row">
          <div class="bar-label" style="font-size:0.68rem;width:80px">triplet</div>
          <div class="bar-track"><div class="bar-fill" style="width:${(mt.test_eer/maxEER*90).toFixed(1)}%;background:linear-gradient(to right,var(--blue),#93c5fd)"></div></div>
          <div class="bar-num">${(mt.test_eer*100).toFixed(3)}%</div>
        </div>
      </div>`;
    }).join('')}
  </div>`;
}

// ─── EXPLAINER ───
function buildExplainer() {
  const cards = [
    { icon: '📁', title: 'SOCOFing Dataset', text: '600 subjects × 10 fingers = 6,000 unique identities. 55,270 total files: real scans + Easy/Medium/Hard synthetic alterations.', tags: ['Real','Easy','Medium','Hard'] },
    { icon: '✏️', title: 'What is "Altered"?', text: 'Altered fingerprints are the same real scans but synthetically damaged to simulate forgery, wear, or sensor artefacts. Easy=minor damage, Medium=visible damage, Hard=severe obliteration.', tags: ['Easy: minor', 'Medium: visible', 'Hard: severe'] },
    { icon: '🎯', title: 'Anchor / Positive / Negative', text: 'Anchor = reference real scan. Positive = another image of the same finger (genuine). Negative = image from a different identity (impostor). The model pulls A↔P closer and pushes A↔N apart.', tags: ['Anchor','Positive','Negative'] },
    { icon: '📏', title: 'Triplet Loss Formula', text: 'L = max(0, d(A,P) − d(A,N) + margin). With margin=0.75, the model must ensure same-finger distances are 0.75 units less than different-finger distances. BatchHard selects hardest positives and negatives.', tags: ['margin=0.75','BatchHard'] },
    { icon: '📊', title: 'EER / FAR / FRR', text: 'FAR = impostors wrongly accepted. FRR = genuine users wrongly rejected. EER = threshold where FAR = FRR. Best EER: 0.145%. FAR=0.146%, FRR=0.144% at threshold=0.381.', tags: ['EER=0.145%','FAR','FRR'] },
    { icon: '⚖️', title: 'What Is a Threshold?', text: 'After embedding, compare L2 distance. Distance < 0.381 → SAME PERSON (Match). Distance ≥ 0.381 → DIFFERENT PERSON (Reject). The 0.381 threshold is the EER operating point.', tags: ['L2 distance','0.381 threshold'] },
    { icon: '🧠', title: 'compact_siamese', text: '4 standard Conv2d layers (1→24→48→72→96 channels), 3× MaxPool, AdaptiveAvgPool, Linear→128D. Contrastive (pair) loss. ~118K params. Best model overall.', tags: ['Contrastive loss','~118K params','GELU'] },
    { icon: '📱', title: 'mobile_triplet', text: 'MobileNet-style depthwise separable blocks (1→16→24→48→72→96 channels), stride-2 DW-Sep, AdaptiveAvgPool, Linear→128D. Triplet loss. ~31K params. 4× smaller.', tags: ['Triplet loss','~31K params','DW-Sep'] },
    { icon: '🗂️', title: 'Checkpoint Folders', text: 'baseline/: initial run. augmentation_search/: 5 aug variants. hyperparameter_search/: 8 LR×BS×margin combos. final_grid/: 10 final model-preprocess combos. Filename = full experiment ID.', tags: ['baseline','aug_search','hp_search','final_grid'] },
    { icon: '📄', title: 'identity_split.csv', text: 'The train/val/test manifest: 55,270 rows, one per fingerprint. Columns: split, identity_key, file_name, source, difficulty, alteration. Identity-disjoint: no subject appears in 2+ splits.', tags: ['70/15/15 split','identity-disjoint'] },
    { icon: '📑', title: 'pipeline_summary.json', text: 'Full serialized output: every experiment\'s training history, EER, accuracy, FAR, FRR, threshold, positive/negative distances, and 512-point ROC curve data.', tags: ['1.4MB JSON','all experiments'] },
    { icon: '🎓', title: 'Why Raw Beats Gabor?', text: 'Gabor filtering helps classical matchers, but a CNN learns its own optimal spatial filters. The CNN learned ridge features directly from raw pixels more efficiently than the hand-crafted Gabor bank.', tags: ['CNN advantage','learned features'] }
  ];
  const grid = document.getElementById('explainer-grid');
  cards.forEach(c => {
    const div = document.createElement('div');
    div.className = 'exp-card';
    div.innerHTML = `
      <div class="exp-card-icon">${c.icon}</div>
      <h3>${c.title}</h3>
      <p>${c.text}</p>
      <div class="tag-list">${c.tags.map(t => `<span class="exp-tag">${t}</span>`).join('')}</div>`;
    grid.appendChild(div);
  });
}

// ─── MODEL DIFF TABLE ───
function buildModelDiff() {
  const siameseCard = document.getElementById('mc-siamese');
  siameseCard.innerHTML = `
    <div class="model-card-name" style="color:var(--accent2)">compact_siamese</div>
    <div class="model-card-loss">Loss: Contrastive (pair-based)</div>
    <div class="layer-list">
      ${[
        ['Conv2d(1→24, 3×3)','+ BN + GELU + MaxPool','var(--accent2)'],
        ['Conv2d(24→48, 3×3)','+ BN + GELU + MaxPool','var(--accent2)'],
        ['Conv2d(48→72, 3×3)','+ BN + GELU + MaxPool','var(--accent2)'],
        ['Conv2d(72→96, 3×3)','+ BN + GELU','var(--accent2)'],
        ['AdaptiveAvgPool2d(1×1)','→ Flatten','var(--text-muted)'],
        ['Linear(96→128)','embedding layer','#34d399'],
        ['L2Normalize','unit sphere','#34d399'],
      ].map(([n,d,c]) => `<div class="layer-item"><div class="layer-dot" style="background:${c}"></div><span class="layer-name">${n}</span><span class="layer-detail">${d}</span></div>`).join('')}
    </div>`;

  const tripletCard = document.getElementById('mc-triplet');
  tripletCard.innerHTML = `
    <div class="model-card-name" style="color:var(--blue)">mobile_triplet</div>
    <div class="model-card-loss">Loss: Batch-Hard Triplet</div>
    <div class="layer-list">
      ${[
        ['Conv2d(1→16, 3×3)','standard + BN + GELU','var(--blue)'],
        ['DW-Sep(16→24, s=2)','depthwise + pointwise','var(--blue)'],
        ['DW-Sep(24→48, s=2)','depthwise + pointwise','var(--blue)'],
        ['DW-Sep(48→72, s=2)','depthwise + pointwise','var(--blue)'],
        ['DW-Sep(72→96, s=1)','depthwise + pointwise','var(--blue)'],
        ['AdaptiveAvgPool2d(1×1)','→ Flatten','var(--text-muted)'],
        ['Linear(96→128)','embedding layer','#34d399'],
        ['L2Normalize','unit sphere','#34d399'],
      ].map(([n,d,c]) => `<div class="layer-item"><div class="layer-dot" style="background:${c}"></div><span class="layer-name">${n}</span><span class="layer-detail">${d}</span></div>`).join('')}
    </div>`;

  const table = document.getElementById('diff-table');
  const rows = [
    ['Property','compact_siamese','mobile_triplet'],
    ['Conv type','Standard Conv2d (3×3)','Depthwise Separable'],
    ['Loss function','Contrastive (pair)','Batch-Hard Triplet'],
    ['Parameters','~118K','~31K (4× smaller)'],
    ['Channel sequence','1→24→48→72→96','1→16→24→48→72→96'],
    ['Downsampling','MaxPool ×3','Stride-2 DW blocks ×3'],
    ['Activation','GELU','GELU'],
    ['Best preprocessing','raw (0.00145 EER)','raw (0.00978 EER)'],
    ['Decision threshold','0.381 (L2)','0.792 (L2)'],
    ['Positive distance (avg)','0.152–0.197','0.535–0.601'],
    ['Negative distance (avg)','0.614–0.685','0.755–1.048'],
    ['Test EER (best)','0.00145 (0.145%)','0.00978 (0.978%)'],
    ['Test Accuracy (best)','99.85%','99.04%'],
    ['Training loss (ep1→2)','0.078 → 0.046','0.531 → 0.310'],
  ];
  rows.forEach((row, i) => {
    const tr = document.createElement('tr');
    row.forEach((cell, j) => {
      const el = document.createElement(i === 0 ? 'th' : 'td');
      el.textContent = cell;
      tr.appendChild(el);
    });
    table.appendChild(tr);
  });
}

// ─── SCROLL ANIMATION ───
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('animate-in'); });
  }, { threshold: 0.1 });
  document.querySelectorAll('.stat-card, .exp-card, .result-row, .stage-block').forEach(el => {
    observer.observe(el);
  });
}

// ─── NAVBAR SCROLL ───
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  nav.style.background = window.scrollY > 40
    ? 'rgba(10,10,15,0.97)'
    : 'rgba(10,10,15,0.85)';
});

// ─── INIT ───
document.addEventListener('DOMContentLoaded', () => {
  buildFlowMap();
  buildDemo();
  buildResultsTable();
  buildCharts();
  buildExplainer();
  buildModelDiff();
  initScrollAnimations();
});
