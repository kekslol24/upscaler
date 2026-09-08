const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const engineSelect = document.getElementById("engineSelect");
const scaleSelect = document.getElementById("scaleSelect");
const gpuSelect = document.getElementById("gpuSelect");
const submitBtn = document.getElementById("submitBtn");
const engineHint = document.getElementById("engineHint");

const progressSection = document.getElementById("progressSection");
const progressMessage = document.getElementById("progressMessage");
const progressFill = document.getElementById("progressFill");

const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");

const compareSection = document.getElementById("compareSection");
const videoBefore = document.getElementById("videoBefore");
const videoAfter = document.getElementById("videoAfter");
const compareRange = document.getElementById("compareRange");
const compareDivider = document.getElementById("compareDivider");
const playPauseBtn = document.getElementById("playPauseBtn");
const downloadLink = document.getElementById("downloadLink");

let engines = [];

async function loadEngines() {
  const res = await fetch("/api/engines");
  engines = await res.json();

  engineSelect.innerHTML = "";
  for (const engine of engines) {
    const opt = document.createElement("option");
    opt.value = engine.name;
    opt.textContent = engine.available ? engine.label : `${engine.label} (unavailable)`;
    opt.disabled = !engine.available;
    if (!engine.available && engine.reason) opt.title = engine.reason;
    engineSelect.appendChild(opt);
  }

  const firstAvailable = engines.find((e) => e.available);
  if (firstAvailable) {
    engineSelect.value = firstAvailable.name;
  } else {
    engineHint.textContent = "No engines available — check the server setup (ffmpeg / engine binary).";
  }
  updateScaleOptions();
  updateEngineHint();
}

function updateEngineHint() {
  const engine = engines.find((e) => e.name === engineSelect.value);
  if (engine && engine.available) {
    engineHint.textContent = engine.description || "";
  } else if (engine) {
    engineHint.textContent = engine.reason || "unavailable";
  }
}

async function loadGpus() {
  const res = await fetch("/api/gpus");
  const { detectable, gpus } = await res.json();

  gpuSelect.innerHTML = "";
  const autoOpt = document.createElement("option");
  autoOpt.value = "";
  autoOpt.textContent = "Auto";
  gpuSelect.appendChild(autoOpt);

  for (const g of gpus) {
    const opt = document.createElement("option");
    opt.value = g.index;
    opt.textContent = `${g.name} (${g.vendor})`;
    gpuSelect.appendChild(opt);
  }

  if (!detectable) {
    gpuSelect.title = "vulkaninfo not found — install vulkan-tools to see specific GPUs";
  } else if (gpus.length === 0) {
    gpuSelect.title = "no Vulkan-capable GPU detected";
  }
}

function updateScaleOptions() {
  const engine = engines.find((e) => e.name === engineSelect.value);
  scaleSelect.innerHTML = "";
  const scales = engine && engine.supported_scales.length ? engine.supported_scales : [2, 4];
  for (const s of scales) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = `${s}x`;
    scaleSelect.appendChild(opt);
  }
}

engineSelect.addEventListener("change", () => {
  updateScaleOptions();
  updateEngineHint();
});

function resetSections() {
  progressSection.hidden = true;
  errorSection.hidden = true;
  compareSection.hidden = true;
  progressFill.style.width = "0%";
}

function showError(msg) {
  errorSection.hidden = false;
  errorMessage.textContent = msg;
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  resetSections();

  const file = fileInput.files[0];
  if (!file) return;

  submitBtn.disabled = true;
  progressSection.hidden = false;
  progressMessage.textContent = "uploading";

  try {
    const formData = new FormData();
    formData.append("file", file);
    const uploadRes = await fetch("/api/upload", { method: "POST", body: formData });
    if (!uploadRes.ok) throw new Error(await uploadRes.text());
    const { job_id } = await uploadRes.json();

    const startRes = await fetch(`/api/jobs/${job_id}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: engineSelect.value,
        scale: Number(scaleSelect.value),
        gpu_id: gpuSelect.value === "" ? null : Number(gpuSelect.value),
      }),
    });
    if (!startRes.ok) throw new Error(await startRes.text());

    pollJob(job_id);
  } catch (err) {
    submitBtn.disabled = false;
    showError(err.message || String(err));
  }
});

function pollJob(jobId) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) throw new Error(await res.text());
      const job = await res.json();

      progressMessage.textContent = job.message || job.status;
      progressFill.style.width = `${job.progress || 0}%`;

      if (job.status === "done") {
        clearInterval(interval);
        submitBtn.disabled = false;
        progressSection.hidden = true;
        showCompare(jobId);
      } else if (job.status === "error") {
        clearInterval(interval);
        submitBtn.disabled = false;
        progressSection.hidden = true;
        showError(job.message || "upscale failed");
      }
    } catch (err) {
      clearInterval(interval);
      submitBtn.disabled = false;
      showError(err.message || String(err));
    }
  }, 1500);
}

function showCompare(jobId) {
  videoBefore.src = `/api/jobs/${jobId}/input`;
  videoAfter.src = `/api/jobs/${jobId}/output`;
  downloadLink.href = `/api/jobs/${jobId}/output`;
  compareSection.hidden = false;
  setSlider(50);
}

function setSlider(pct) {
  compareRange.value = pct;
  videoAfter.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  compareDivider.style.left = `${pct}%`;
}

compareRange.addEventListener("input", (e) => setSlider(Number(e.target.value)));

playPauseBtn.addEventListener("click", () => {
  if (videoBefore.paused) {
    videoBefore.play();
    videoAfter.play();
    playPauseBtn.textContent = "Pause";
  } else {
    videoBefore.pause();
    videoAfter.pause();
    playPauseBtn.textContent = "Play";
  }
});

// videoBefore drives playback; keep videoAfter's clock from drifting off it.
videoBefore.addEventListener("timeupdate", () => {
  if (Math.abs(videoBefore.currentTime - videoAfter.currentTime) > 0.15) {
    videoAfter.currentTime = videoBefore.currentTime;
  }
});

loadEngines();
loadGpus();
