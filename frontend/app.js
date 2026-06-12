/* ─── cGAN Studio — Frontend Logic ──────────────────────────────────
   Handles: class selection, generation API calls, gallery loading,
            slider UX, lightbox, and download helpers.
─────────────────────────────────────────────────────────────────── */

const API = '';  // same-origin

// ── State ──────────────────────────────────────────────────────────
let selectedClass = 1;   // default: automobile
let generatedImages = []; // [{src, label}]

// ── DOM refs ───────────────────────────────────────────────────────
const classGrid      = document.getElementById('class-grid');
const numImagesSlider = document.getElementById('num-images');
const countDisplay   = document.getElementById('count-display');
const generateBtn    = document.getElementById('generate-btn');
const imageGrid      = document.getElementById('image-grid');
const emptyState     = document.getElementById('empty-state');
const outputHeader   = document.getElementById('output-header');
const outputMeta     = document.getElementById('output-meta');
const downloadAllBtn = document.getElementById('download-all-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingSub     = document.getElementById('loading-sub');
const galleryGrid    = document.getElementById('gallery-grid');
const lightbox       = document.getElementById('lightbox');
const lightboxImg    = document.getElementById('lightbox-img');
const lightboxCaption= document.getElementById('lightbox-caption');
const lightboxClose  = document.getElementById('lightbox-close');
const lightboxDl     = document.getElementById('lightbox-dl');

// ── Tab navigation ─────────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');
    if (tab === 'gallery') loadGallery();
  });
});

// ── Load classes from API ──────────────────────────────────────────
async function loadClasses() {
  const res = await fetch(`${API}/api/classes`);
  const classes = await res.json();

  classGrid.innerHTML = '';
  classes.forEach(cls => {
    const chip = document.createElement('button');
    chip.className = 'class-chip' + (cls.id === selectedClass ? ' selected' : '');
    chip.id = `class-chip-${cls.id}`;
    chip.innerHTML = `
      <span class="chip-emoji">${cls.emoji}</span>
      <span class="chip-label">${cls.name}</span>
    `;
    chip.addEventListener('click', () => {
      document.querySelectorAll('.class-chip').forEach(c => c.classList.remove('selected'));
      chip.classList.add('selected');
      selectedClass = cls.id;
    });
    classGrid.appendChild(chip);
  });
}

// ── Slider ─────────────────────────────────────────────────────────
function updateSlider() {
  const val = numImagesSlider.value;
  const max = numImagesSlider.max;
  const pct = ((val - 1) / (max - 1)) * 100;
  numImagesSlider.style.setProperty('--pct', `${pct}%`);
  countDisplay.textContent = val;
}
numImagesSlider.addEventListener('input', updateSlider);
updateSlider();

// ── Generate ───────────────────────────────────────────────────────
generateBtn.addEventListener('click', async () => {
  const numImages = parseInt(numImagesSlider.value);
  generateBtn.disabled = true;
  showLoading(`Generating ${numImages} image${numImages > 1 ? 's' : ''} through the Generator…`);

  try {
    const res = await fetch(`${API}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ class_id: selectedClass, num_images: numImages })
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    hideLoading();

    // Prepend to generatedImages
    const newItems = data.images.map((src, i) => ({
      src,
      label: `${data.class_emoji} ${data.class_name} #${generatedImages.length + i + 1}`
    }));
    generatedImages = [...newItems, ...generatedImages];

    renderImageGrid();
    outputHeader.style.display = 'flex';
    const hiresTag = data.hires
      ? ` <span style="color:var(--accent2);font-weight:700">✨ Hi-Res (64×64)</span>`
      : ` <span style="color:var(--muted);font-size:0.78rem">(32×32 model)</span>`;
    outputMeta.innerHTML = `<strong>${data.images.length}</strong> ${data.class_emoji} <strong>${data.class_name}</strong> image${data.images.length > 1 ? 's' : ''} generated${hiresTag}`;

  } catch (err) {
    hideLoading();
    alert(`❌ Generation failed: ${err.message}`);
  } finally {
    generateBtn.disabled = false;
  }
});

// ── Render image grid ──────────────────────────────────────────────
function renderImageGrid() {
  imageGrid.innerHTML = '';

  if (generatedImages.length === 0) {
    imageGrid.appendChild(emptyState);
    return;
  }

  generatedImages.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'img-card';
    card.style.animationDelay = `${idx * 60}ms`;
    card.innerHTML = `
      <img src="${item.src}" alt="${item.label}" loading="lazy" />
      <div class="img-card-overlay">
        <span class="img-card-label">${item.label}</span>
      </div>
      <button class="img-dl-btn" title="Download" data-idx="${idx}">⬇</button>
    `;

    card.querySelector('.img-dl-btn').addEventListener('click', e => {
      e.stopPropagation();
      downloadImage(item.src, `cgan_${item.label.replace(/[^a-z0-9]/gi,'_')}.png`);
    });

    card.addEventListener('click', () => openLightbox(item));
    imageGrid.appendChild(card);
  });
}

// ── Download helpers ───────────────────────────────────────────────
function downloadImage(src, filename) {
  const a = document.createElement('a');
  a.href = src;
  a.download = filename;
  a.click();
}

downloadAllBtn.addEventListener('click', () => {
  generatedImages.forEach((item, i) => {
    setTimeout(() => downloadImage(item.src, `cgan_${i+1}_${item.label.replace(/[^a-z0-9]/gi,'_')}.png`), i * 200);
  });
});

// ── Lightbox ───────────────────────────────────────────────────────
function openLightbox({ src, label }) {
  lightboxImg.src = src;
  lightboxCaption.textContent = label;
  lightboxDl.onclick = () => downloadImage(src, `cgan_${label.replace(/[^a-z0-9]/gi,'_')}.png`);
  lightbox.classList.add('visible');
}
function closeLightbox() { lightbox.classList.remove('visible'); }
lightboxClose.addEventListener('click', closeLightbox);
lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── Loading overlay ────────────────────────────────────────────────
function showLoading(msg = 'Running inference…') {
  loadingSub.textContent = msg;
  loadingOverlay.classList.add('visible');
}
function hideLoading() { loadingOverlay.classList.remove('visible'); }

// ── Gallery ────────────────────────────────────────────────────────
let galleryLoaded = false;
async function loadGallery() {
  if (galleryLoaded) return;
  try {
    const res = await fetch(`${API}/api/gallery`);
    const items = await res.json();
    galleryGrid.innerHTML = '';

    if (items.length === 0) {
      galleryGrid.innerHTML = `<div class="loading-state">No training samples found in outputs/generated_images/</div>`;
      return;
    }

    items.reverse().forEach(item => {
      const card = document.createElement('div');
      card.className = 'gallery-card';
      card.innerHTML = `
        <img src="${item.path}" alt="${item.label}" loading="lazy" />
        <div class="gallery-card-label">${item.label}</div>
      `;
      card.addEventListener('click', () => {
        lightboxImg.src = item.path;
        lightboxCaption.textContent = item.label;
        lightboxDl.onclick = () => {
          const a = document.createElement('a');
          a.href = item.path; a.download = item.label + '.png'; a.click();
        };
        lightbox.classList.add('visible');
      });
      galleryGrid.appendChild(card);
    });
    galleryLoaded = true;
  } catch (err) {
    galleryGrid.innerHTML = `<div class="loading-state">Failed to load gallery: ${err.message}</div>`;
  }
}

// ── Init ───────────────────────────────────────────────────────────
loadClasses();
