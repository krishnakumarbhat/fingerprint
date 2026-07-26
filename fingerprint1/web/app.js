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

// ─── DEMO ───
let activeCase = null;
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
  showCase(TEST_CASES[0]);
}

function showCase(tc) {
  activeCase = tc;
  document.querySelectorAll('.tc-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`btn-${tc.id}`);
  if (btn) btn.classList.add('active');

  const colorA = tc.fp1.source === 'real' ? '#7c6af7' : (tc.fp1.difficulty === 'Easy' ? '#34d399' : tc.fp1.difficulty === 'Hard' ? '#fb923c' : '#60a5fa');
  const colorB = tc.fp2.source === 'real' && tc.fp2.emoji === '🔴' ? '#f87171' : (tc.fp2.difficulty === 'Easy' ? '#34d399' : tc.fp2.difficulty === 'Hard' ? '#fb923c' : '#7c6af7');
  const seedA = tc.id.charCodeAt(2) * 7 + 1;
  const seedB = tc.id.charCodeAt(2) * 13 + (tc.verdict === 'MATCH' ? 3 : 99);

  document.getElementById('demo-fingerprints').innerHTML = `
    <div class="fp-slot">
      <div class="fp-visual"><canvas id="canvas-a-${tc.id}"></canvas></div>
      <div class="fp-label">${tc.fp1.label}</div>
      <div class="fp-meta">${tc.fp1.meta}</div>
      <div class="fp-meta" style="color:var(--text-muted)">Source: ${tc.fp1.source} | ${tc.fp1.difficulty}</div>
    </div>
    <div class="vs-divider">
      <div style="margin-bottom:0.4rem">${tc.dist.toFixed(3)}</div>
      <div class="dist-arrow"></div>
      <div style="margin-top:0.4rem;font-size:0.7rem;color:var(--text-muted)">L2 distance</div>
    </div>
    <div class="fp-slot">
      <div class="fp-visual"><canvas id="canvas-b-${tc.id}"></canvas></div>
      <div class="fp-label">${tc.fp2.label}</div>
      <div class="fp-meta">${tc.fp2.meta}</div>
      <div class="fp-meta" style="color:var(--text-muted)">Source: ${tc.fp2.source} | ${tc.fp2.difficulty}</div>
    </div>`;

  setTimeout(() => {
    const ca = document.getElementById(`canvas-a-${tc.id}`);
    const cb = document.getElementById(`canvas-b-${tc.id}`);
    if (ca) drawFP(ca, seedA, colorA);
    if (cb) drawFP(cb, seedB, tc.verdict === 'MATCH' ? colorA : '#f87171');
  }, 10);

  const distColor = tc.dist < tc.threshold ? 'val-green' : 'val-red';
  const verdictClass = tc.verdict === 'MATCH' ? 'match' : 'nomatch';
  const icon = tc.verdict === 'MATCH' ? '✅' : '❌';
  const gapToThreshold = (tc.threshold - tc.dist).toFixed(3);

  document.getElementById('demo-analysis').innerHTML = `
    <div class="analysis-card">
      <div class="analysis-card-label">L2 Distance</div>
      <div class="analysis-card-val ${distColor}">${tc.dist.toFixed(3)}</div>
      <div class="analysis-card-sub">embedding space</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Threshold</div>
      <div class="analysis-card-val val-yellow">${tc.threshold.toFixed(3)}</div>
      <div class="analysis-card-sub">decision boundary</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Gap to Threshold</div>
      <div class="analysis-card-val ${tc.verdict==='MATCH'?'val-green':'val-red'}">${gapToThreshold}</div>
      <div class="analysis-card-sub">${tc.verdict==='MATCH'?'safe margin':'exceeded by'}</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Positive Dist</div>
      <div class="analysis-card-val val-accent">${tc.pos_dist.toFixed(3)}</div>
      <div class="analysis-card-sub">same-finger avg</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Negative Dist</div>
      <div class="analysis-card-val val-blue">${tc.neg_dist.toFixed(3)}</div>
      <div class="analysis-card-sub">diff-finger avg</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Margin</div>
      <div class="analysis-card-val val-yellow">${tc.margin}</div>
      <div class="analysis-card-sub">contrastive loss</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Model EER</div>
      <div class="analysis-card-val val-green">${(tc.eer*100).toFixed(3)}%</div>
      <div class="analysis-card-sub">equal error rate</div>
    </div>
    <div class="analysis-card">
      <div class="analysis-card-label">Contrastive Loss</div>
      <div class="analysis-card-val val-accent">${tc.contrastive_loss.toFixed(4)}</div>
      <div class="analysis-card-sub">training signal</div>
    </div>
    <div class="verdict-banner ${verdictClass}" style="grid-column:1/-1">
      <span class="verdict-icon">${icon}</span>
      <div>
        <div>${tc.verdict === 'MATCH' ? 'SAME PERSON — Fingerprints match!' : 'DIFFERENT PERSON — Impostor rejected!'}</div>
        <div class="verdict-details">${tc.details}</div>
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
