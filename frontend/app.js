const $ = (id) => document.getElementById(id);

const fileInput = $("fileInput");
const sourceInfo = $("sourceInfo");
const presetSelect = $("presetSelect");
const presetHint = $("presetHint");
const engineSelect = $("engineSelect");
const scaleSelect = $("scaleSelect");
const gpuSelect = $("gpuSelect");
const engineHint = $("engineHint");
const deinterlaceInput = $("deinterlaceInput");
const denoiseSelect = $("denoiseSelect");
const sharpenInput = $("sharpenInput");
const interpFactorSelect = $("interpFactorSelect");
const interpModeSelect = $("interpModeSelect");
const interpEngineSelect = $("interpEngineSelect");
const interpHint = $("interpHint");
const codecSelect = $("codecSelect");
const formatSelect = $("formatSelect");
const subtitlesSelect = $("subtitlesSelect");
const encoderSelect = $("encoderSelect");
const qualityInput = $("qualityInput");
const encoderHint = $("encoderHint");
const previewStartInput = $("previewStartInput");
const previewLengthInput = $("previewLengthInput");
const previewBtn = $("previewBtn");
const renderBtn = $("renderBtn");
const renderAllBtn = $("renderAllBtn");

const progressSection = $("progressSection");
const progressMessage = $("progressMessage");
const progressFill = $("progressFill");
const cancelBtn = $("cancelBtn");
const pauseBtn = $("pauseBtn");

const errorSection = $("errorSection");
const errorMessage = $("errorMessage");

const compareSection = $("compareSection");
const compareTitle = $("compareTitle");
const videoBefore = $("videoBefore");
const videoAfter = $("videoAfter");
const compareRange = $("compareRange");
const compareDivider = $("compareDivider");
const playPauseBtn = $("playPauseBtn");
const downloadLink = $("downloadLink");
const downloadAltLink = $("downloadAltLink");
const outputPath = $("outputPath");
const jobsSection = $("jobsSection");
const jobsBody = $("jobsBody");
const clearJobsBtn = $("clearJobsBtn");

//: statuses the backend allows deleting (active renders must be cancelled)
const REMOVABLE_STATUSES = new Set(["uploaded", "done", "error", "cancelled"]);

let engines = [];
let presets = [];
let encoders = [];
let interpolators = [];
const NO_UPSCALE = "none";
let currentJob = null;   // the uploaded source we render from
let uploading = false;   // a multi-file upload is in progress
let activeJobId = null;  // the job currently being polled
let activeJobStatus = null;
let pauseResumeSupported = false;
let pollTimer = null;

// -- catalogue loading ---------------------------------------------------

async function fetchJson(url, init) {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function loadEngines() {
  engines = await fetchJson("/api/engines");
  engineSelect.innerHTML = "";
  for (const engine of engines) {
    const opt = document.createElement("option");
    opt.value = engine.name;
    opt.textContent = engine.available ? engine.label : `${engine.label} (unavailable)`;
    opt.disabled = !engine.available;
    if (!engine.available && engine.reason) opt.title = engine.reason;
    engineSelect.appendChild(opt);
  }
  const noneOpt = document.createElement("option");
  noneOpt.value = NO_UPSCALE;
  noneOpt.textContent = "No upscaling (frame interpolation only)";
  engineSelect.appendChild(noneOpt);
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
  if (engineSelect.value === NO_UPSCALE) {
    engineHint.textContent = "Frames pass through at their original size; turn on interpolation below.";
  } else if (engine && engine.available) {
    engineHint.textContent = engine.description || "";
  } else if (engine) {
    engineHint.textContent = engine.reason || "unavailable";
  }
}

async function loadInterpolators() {
  const data = await fetchJson("/api/interpolators");
  interpolators = data.interpolators;
  interpFactorSelect.innerHTML = "";
  const off = document.createElement("option");
  off.value = "0";
  off.textContent = "Off";
  interpFactorSelect.appendChild(off);
  for (const f of data.factors) {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = `${f}x frames`;
    interpFactorSelect.appendChild(opt);
  }
  interpEngineSelect.innerHTML = "";
  for (const i of interpolators) {
    const opt = document.createElement("option");
    opt.value = i.name;
    opt.textContent = i.available ? i.label : `${i.label} (unavailable)`;
    opt.disabled = !i.available;
    if (!i.available && i.reason) opt.title = i.reason;
    interpEngineSelect.appendChild(opt);
  }
  const firstAvailable = interpolators.find((i) => i.available);
  if (firstAvailable) interpEngineSelect.value = firstAvailable.name;
  updateInterpHint();
}

function updateInterpHint() {
  const on = Number(interpFactorSelect.value) > 0;
  interpModeSelect.disabled = !on;
  interpEngineSelect.disabled = !on;
  const interp = interpolators.find((i) => i.name === interpEngineSelect.value);
  if (!on) {
    interpHint.textContent = interpolators.some((i) => i.available)
      ? ""
      : (interp && interp.reason) || "no frame interpolation engine installed";
  } else if (interp && !interp.available) {
    interpHint.textContent = interp.reason || "unavailable";
  } else if (interp) {
    interpHint.textContent = interp.description || "";
  }
}

interpFactorSelect.addEventListener("change", () => { updateInterpHint(); markCustom(); });
interpEngineSelect.addEventListener("change", () => { updateInterpHint(); markCustom(); });
interpModeSelect.addEventListener("change", markCustom);

async function loadGpus() {
  const { detectable, gpus } = await fetchJson("/api/gpus");
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
    gpuSelect.title = "GPU detection unavailable — install vulkan-tools or use an NVIDIA GPU with nvidia-smi";
  } else if (gpus.length === 0) {
    gpuSelect.title = "no GPU detected";
  }
}

async function loadEncoders() {
  encoders = await fetchJson("/api/encoders");
  updateEncoderOptions();
}

function updateEncoderOptions() {
  const codec = codecSelect.value;
  const previous = encoderSelect.value;
  encoderSelect.innerHTML = "";
  const autoOpt = document.createElement("option");
  autoOpt.value = "auto";
  autoOpt.textContent = "Auto (GPU if available)";
  encoderSelect.appendChild(autoOpt);
  const matching = encoders.filter((e) => e.codec === codec);
  for (const e of matching) {
    const opt = document.createElement("option");
    opt.value = e.name;
    opt.textContent = e.label;
    encoderSelect.appendChild(opt);
  }
  encoderSelect.value = matching.some((e) => e.name === previous) ? previous : "auto";
  const best = matching[0];
  encoderHint.textContent = best
    ? `Auto uses ${best.label}.`
    : `No working ${codec.toUpperCase()} encoder found on this machine.`;
}

async function loadPresets() {
  presets = await fetchJson("/api/presets");
  presetSelect.innerHTML = "";
  for (const p of presets) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.label;
    presetSelect.appendChild(opt);
  }
  const custom = document.createElement("option");
  custom.value = "";
  custom.textContent = "Custom";
  presetSelect.appendChild(custom);
  if (presets.length) applyPreset(presets[0]);
}

function applyPreset(preset) {
  presetSelect.value = preset.name;
  presetHint.textContent = preset.description;
  const engine = preset.engines.find(
    (name) => name === NO_UPSCALE || engines.some((e) => e.name === name && e.available),
  );
  if (engine) engineSelect.value = engine;
  updateScaleOptions();
  if ([...scaleSelect.options].some((o) => Number(o.value) === preset.scale)) {
    scaleSelect.value = preset.scale;
  }
  updateEngineHint();
  const f = preset.filters || {};
  deinterlaceInput.checked = !!f.deinterlace;
  denoiseSelect.value = String(f.denoise || 0);
  sharpenInput.checked = !!f.sharpen;
  const interp = preset.interpolate || {};
  interpFactorSelect.value = String(interp.factor || 0);
  if (interp.mode) interpModeSelect.value = interp.mode;
  if (interp.engine) {
    const preferred = interpolators.find((i) => i.name === interp.engine && i.available);
    const fallback = interpolators.find((i) => i.available);
    if (preferred || fallback) interpEngineSelect.value = (preferred || fallback).name;
  }
  updateInterpHint();
  const enc = preset.encode || {};
  codecSelect.value = enc.codec || "hevc";
  updateEncoderOptions();
  encoderSelect.value = enc.encoder || "auto";
  qualityInput.value = enc.quality || 18;
  formatSelect.value = enc.format || "mp4";
  subtitlesSelect.value = enc.subtitles || "copy";
}

function markCustom() {
  presetSelect.value = "";
  presetHint.textContent = "";
}

function updateScaleOptions() {
  const engine = engines.find((e) => e.name === engineSelect.value);
  const previous = scaleSelect.value;
  scaleSelect.innerHTML = "";
  let scales = engine && engine.supported_scales.length ? engine.supported_scales : [2, 4];
  if (engineSelect.value === NO_UPSCALE) scales = [1];
  scaleSelect.disabled = engineSelect.value === NO_UPSCALE;
  for (const s of scales) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = `${s}x`;
    scaleSelect.appendChild(opt);
  }
  if (scales.map(String).includes(previous)) scaleSelect.value = previous;
}

presetSelect.addEventListener("change", () => {
  const preset = presets.find((p) => p.name === presetSelect.value);
  if (preset) applyPreset(preset);
  else presetHint.textContent = "";
});

engineSelect.addEventListener("change", () => {
  updateScaleOptions();
  updateEngineHint();
  markCustom();
});
codecSelect.addEventListener("change", () => {
  updateEncoderOptions();
  markCustom();
});
for (const el of [scaleSelect, deinterlaceInput, denoiseSelect, sharpenInput, encoderSelect, qualityInput, subtitlesSelect]) {
  el.addEventListener("change", markCustom);
}

// -- source upload -------------------------------------------------------

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function setSource(job) {
  currentJob = job;
  sourceInfo.textContent = `${job.filename} — ${job.width}×${job.height}, ${formatTime(job.duration)}`;
  previewStartInput.max = Math.max(0, job.duration - 1).toFixed(1);
  if (Number(previewStartInput.value) > job.duration) previewStartInput.value = 0;
  previewBtn.disabled = false;
  renderBtn.disabled = false;
}

function forgetSource(jobId) {
  if (!currentJob || currentJob.id !== jobId) return;
  currentJob = null;
  previewBtn.disabled = true;
  renderBtn.disabled = true;
  sourceInfo.textContent = "Pick one or more files — they upload straight away so you can preview settings before committing.";
}

const UPLOAD_CHUNK = 8 * 1024 * 1024;

function newUploadId() {
  if (crypto.randomUUID) return crypto.randomUUID().replace(/-/g, "");
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function uploadSource(file, label = "") {
  const uploadId = newUploadId();
  const total = Math.max(1, Math.ceil(file.size / UPLOAD_CHUNK));
  const prefix = label ? `${label}: ` : "";
  for (let i = 0; i < total; i++) {
    const chunk = file.slice(i * UPLOAD_CHUNK, (i + 1) * UPLOAD_CHUNK);
    const res = await fetch(
      `/api/upload/chunk?upload_id=${uploadId}&index=${i}`,
      { method: "POST", body: chunk },
    );
    if (!res.ok) throw new Error((await res.text()) || `upload chunk ${i} failed`);
    const pct = Math.round(((i + 1) / total) * 100);
    sourceInfo.textContent = `uploading ${prefix}${file.name}… ${pct}%`;
  }
  return fetchJson(
    `/api/upload/complete?upload_id=${uploadId}&filename=${encodeURIComponent(file.name)}`,
    { method: "POST" },
  );
}

fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files);
  if (!files.length) return;
  resetSections();
  previewBtn.disabled = true;
  renderBtn.disabled = true;
  renderAllBtn.disabled = true;
  let last = null;
  const errors = [];
  uploading = true;
  try {
    for (let i = 0; i < files.length; i++) {
      const label = `${i + 1}/${files.length}`;
      sourceInfo.textContent = `uploading ${label}: ${files[i].name}…`;
      try {
        last = await uploadSource(files[i], label);
      } catch (err) {
        errors.push(`${files[i].name}: ${err.message || err}`);
      }
    }
  } finally {
    uploading = false;
  }
  fileInput.value = "";  // allow selecting the same files again
  if (last) setSource(last);
  loadJobs();
  if (files.length > 1 && !errors.length) {
    sourceInfo.textContent =
      `${files.length} sources uploaded — choose settings, then “Render all uploaded”.`;
  }
  if (errors.length) showError(errors.join("\n"));
});

// -- rendering -----------------------------------------------------------

function collectOptions(preview) {
  const options = {
    filters: {
      deinterlace: deinterlaceInput.checked,
      denoise: Number(denoiseSelect.value),
      sharpen: sharpenInput.checked,
    },
    encode: {
      codec: codecSelect.value,
      format: formatSelect.value,
      subtitles: subtitlesSelect.value,
      encoder: encoderSelect.value,
      quality: Number(qualityInput.value),
    },
  };
  if (Number(interpFactorSelect.value) > 0) {
    options.interpolate = {
      engine: interpEngineSelect.value,
      factor: Number(interpFactorSelect.value),
      mode: interpModeSelect.value,
    };
  }
  if (preview) {
    options.clip = {
      start: Number(previewStartInput.value) || 0,
      duration: Number(previewLengthInput.value) || 5,
    };
  }
  return options;
}

async function startRender(preview) {
  if (!currentJob) return;
  resetSections();
  setBusy(true);
  progressSection.hidden = false;
  progressMessage.textContent = "starting";
  try {
    await fetchJson(`/api/jobs/${currentJob.id}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: engineSelect.value,
        scale: Number(scaleSelect.value),
        gpu_id: gpuSelect.value === "" ? null : Number(gpuSelect.value),
        options: collectOptions(preview),
      }),
    });
    pollJob(currentJob.id);
  } catch (err) {
    setBusy(false);
    progressSection.hidden = true;
    showError(err.message || String(err));
  }
}

async function renderAllUploaded() {
  const body = {
    engine: engineSelect.value,
    scale: Number(scaleSelect.value),
    gpu_id: gpuSelect.value === "" ? null : Number(gpuSelect.value),
    options: collectOptions(false),
  };
  let jobs;
  try {
    jobs = await fetchJson("/api/jobs");
  } catch (err) {
    showError(err.message || String(err));
    return;
  }
  const pending = jobs.filter((job) => job.status === "uploaded");
  if (!pending.length) {
    showError("No uploaded sources to render.");
    return;
  }
  const errors = [];
  for (const job of pending) {
    try {
      await fetchJson(`/api/jobs/${job.id}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (err) {
      errors.push(`${job.filename}: ${err.message || err}`);
    }
  }
  loadJobs();
  if (errors.length) showError(errors.join("\n"));
}

renderAllBtn.addEventListener("click", renderAllUploaded);
previewBtn.addEventListener("click", () => startRender(true));
renderBtn.addEventListener("click", () => startRender(false));

clearJobsBtn.addEventListener("click", async () => {
  if (!confirm("Remove all finished and uploaded jobs and delete their files?")) return;
  clearJobsBtn.disabled = true;
  try {
    await fetchJson("/api/jobs", { method: "DELETE" });
    if (currentJob && REMOVABLE_STATUSES.has(currentJob.status)) {
      currentJob = null;
      previewBtn.disabled = true;
      renderBtn.disabled = true;
      sourceInfo.textContent = "Pick one or more files — they upload straight away so you can preview settings before committing.";
    }
    loadJobs();
  } catch (err) {
    showError(err.message || String(err));
    clearJobsBtn.disabled = false;
  }
});

cancelBtn.addEventListener("click", async () => {
  if (!activeJobId) return;
  cancelBtn.disabled = true;
  try {
    await fetchJson(`/api/jobs/${activeJobId}/cancel`, { method: "POST" });
  } catch (err) {
    showError(err.message || String(err));
  }
});

pauseBtn.addEventListener("click", async () => {
  if (!activeJobId) return;
  const resume = activeJobStatus === "paused";
  pauseBtn.disabled = true;
  try {
    const job = await fetchJson(
      `/api/jobs/${activeJobId}/${resume ? "resume" : "pause"}`,
      { method: "POST" },
    );
    activeJobStatus = job.status;
    progressMessage.textContent = job.message || job.status;
    progressFill.style.width = `${job.progress || 0}%`;
    pauseBtn.textContent = job.status === "paused" ? "Resume" : "Pause";
    loadJobs();
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    pauseBtn.disabled = false;
  }
});

function setBusy(busy) {
  previewBtn.disabled = busy || !currentJob;
  renderBtn.disabled = busy || !currentJob;
  cancelBtn.disabled = !busy;
  pauseBtn.disabled = !busy;
}

function resetSections() {
  clearInterval(pollTimer);
  activeJobId = null;
  activeJobStatus = null;
  setBusy(false);
  cancelBtn.disabled = true;
  pauseBtn.hidden = true;
  progressSection.hidden = true;
  errorSection.hidden = true;
  compareSection.hidden = true;
  progressFill.style.width = "0%";
}

function showError(msg) {
  errorSection.hidden = false;
  errorMessage.textContent = msg;
}

function pollJob(jobId) {
  activeJobId = jobId;
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const job = await fetchJson(`/api/jobs/${jobId}`);
      activeJobStatus = job.status;
      progressMessage.textContent = job.message || job.status;
      progressFill.style.width = `${job.progress || 0}%`;

      if (job.status === "queued" || job.status === "running" || job.status === "paused") {
        pauseBtn.hidden = !pauseResumeSupported || job.pausable === false;
        pauseBtn.textContent = job.status === "paused" ? "Resume" : "Pause";
        pauseBtn.disabled = false;
        return;
      }

      clearInterval(pollTimer);
      activeJobId = null;
      activeJobStatus = null;
      setBusy(false);
      pauseBtn.hidden = true;
      progressSection.hidden = true;
      if (job.status === "done") {
        showCompare(job);
      } else if (job.status === "cancelled") {
        showError("render cancelled");
      } else {
        showError(job.message || "upscale failed");
      }
      loadJobs();
    } catch (err) {
      clearInterval(pollTimer);
      activeJobId = null;
      setBusy(false);
      showError(err.message || String(err));
    }
  }, 1500);
}

// -- compare view --------------------------------------------------------

function describeRender(job) {
  const clip = job.options && job.options.clip;
  const interp = job.options && job.options.interpolate;
  const parts = [];
  if (job.scale > 1) parts.push(`${job.scale}x`);
  if (interp && interp.engine) parts.push(`${interp.factor}x ${interp.mode === "slowmo" ? "slow-mo" : "fps"}`);
  parts.push(clip ? `preview ${formatTime(clip.start)}–${formatTime(clip.start + clip.duration)}` : "full render");
  return parts.join(" · ");
}

function showCompare(job, compareOverride) {
  const compare = compareOverride === undefined
    ? Boolean(job.options && job.options.clip && job.before_ready)
    : Boolean(compareOverride && job.before_ready);
  compareTitle.textContent = compare
    ? `Before / After — ${describeRender(job)}`
    : `Result — ${describeRender(job)}`;
  videoAfter.src = `/api/jobs/${job.id}/output`;
  outputPath.textContent = job.output_path ? `Saved to ${job.output_path}` : "";
  if (compare) {
    videoBefore.hidden = false;
    compareDivider.hidden = false;
    compareRange.hidden = false;
    videoAfter.muted = true;
    videoBefore.src = `/api/jobs/${job.id}/before`;
    setSlider(50);
  } else {
    // A full render has no source-side comparison; show the result on its own.
    videoBefore.hidden = true;
    videoBefore.removeAttribute("src");
    compareDivider.hidden = true;
    compareRange.hidden = true;
    videoAfter.muted = false;
    videoAfter.style.clipPath = "none";
  }
  const mkvPrimary = job.output_format === "mkv" && job.has_subtitle_output;
  const subFormat = (job.options && job.options.encode && job.options.encode.subtitles) || "copy";
  const subLabel = subFormat === "srt" ? "SRT" : subFormat === "ass" ? "ASS" : "";
  const subSuffix = subLabel ? ` (${subLabel} subtitles)` : " (with subtitles)";
  downloadLink.href = `/api/jobs/${job.id}/output?download=1`;
  downloadLink.download = job.download_filename || job.output_filename;
  downloadLink.textContent = `Download ${job.download_filename || job.output_filename}`
    + (mkvPrimary ? subSuffix : "");
  if (mkvPrimary) {
    downloadAltLink.hidden = false;
    downloadAltLink.href = `/api/jobs/${job.id}/output?download=1&mp4=1`;
    downloadAltLink.download = job.output_filename;
    downloadAltLink.textContent = `Download ${job.output_filename} (browser MP4)`;
  } else if (job.has_subtitle_output) {
    downloadAltLink.hidden = false;
    downloadAltLink.href = `/api/jobs/${job.id}/output?download=1&mkv=1`;
    downloadAltLink.download = job.subtitle_output_filename;
    downloadAltLink.textContent =
      `Download ${job.subtitle_output_filename}${subSuffix}`;
  } else {
    downloadAltLink.hidden = true;
  }
  compareSection.hidden = false;
  playPauseBtn.textContent = "Play";
  if (compare) setSlider(50);
  compareSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setSlider(pct) {
  compareRange.value = pct;
  videoAfter.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  compareDivider.style.left = `${pct}%`;
}

compareRange.addEventListener("input", (e) => setSlider(Number(e.target.value)));

playPauseBtn.addEventListener("click", () => {
  const primary = videoBefore.hidden ? videoAfter : videoBefore;
  if (primary.paused) {
    primary.play();
    if (primary === videoBefore) videoAfter.play();
    playPauseBtn.textContent = "Pause";
  } else {
    primary.pause();
    if (primary === videoBefore) videoAfter.pause();
    playPauseBtn.textContent = "Play";
  }
});

// videoBefore drives playback; keep videoAfter's clock from drifting off it.
videoBefore.addEventListener("timeupdate", () => {
  if (Math.abs(videoBefore.currentTime - videoAfter.currentTime) > 0.15) {
    videoAfter.currentTime = videoBefore.currentTime;
  }
});

// -- recent jobs ---------------------------------------------------------

async function loadJobs() {
  let jobs;
  try {
    jobs = await fetchJson("/api/jobs");
  } catch {
    return;
  }
  jobsBody.innerHTML = "";
  jobsSection.hidden = jobs.length === 0;
  const removable = jobs.filter((job) => REMOVABLE_STATUSES.has(job.status));
  clearJobsBtn.hidden = removable.length === 0;
  clearJobsBtn.disabled = uploading;
  renderAllBtn.disabled = uploading || !jobs.some((job) => job.status === "uploaded");
  for (const job of jobs) {
    const tr = document.createElement("tr");
    const render = job.scale ? describeRender(job) : "not rendered";
    const status = (job.status === "running" || job.status === "paused")
      ? `${job.status} ${job.progress}%`
      : job.status;
    const gpu = job.gpu_id === null || job.gpu_id === undefined ? "—" : job.gpu_id;
    tr.innerHTML = `<td title="${job.filename}">${job.filename}</td><td>${render}</td><td>${gpu}</td><td>${status}</td><td class="job-actions"></td>`;
    const actions = tr.querySelector(".job-actions");

    const useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.className = "small secondary";
    useBtn.textContent = "Use as source";
    useBtn.addEventListener("click", () => {
      resetSections();
      setSource(job);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    actions.appendChild(useBtn);

    if (job.has_output && job.status === "done") {
      const viewBtn = document.createElement("button");
      viewBtn.type = "button";
      viewBtn.className = "small";
      viewBtn.textContent = "View";
      viewBtn.addEventListener("click", () => {
        resetSections();
        showCompare(job);
      });
      actions.appendChild(viewBtn);
    }
    if (pauseResumeSupported && job.pausable !== false
        && (job.status === "running" || job.status === "paused" || job.status === "queued")) {
      const pauseBtn = document.createElement("button");
      pauseBtn.type = "button";
      pauseBtn.className = "small secondary";
      pauseBtn.textContent = job.status === "paused" ? "Resume" : "Pause";
      pauseBtn.addEventListener("click", async () => {
        pauseBtn.disabled = true;
        const action = job.status === "paused" ? "resume" : "pause";
        try {
          await fetchJson(`/api/jobs/${job.id}/${action}`, { method: "POST" });
          loadJobs();
        } catch (err) {
          showError(err.message || String(err));
        } finally {
          pauseBtn.disabled = false;
        }
      });
      actions.appendChild(pauseBtn);
    }
    if (job.status === "running" || job.status === "paused" || job.status === "queued"
        || (job.status === "done" && job.has_output)) {
      const watchBtn = document.createElement("button");
      watchBtn.type = "button";
      watchBtn.className = "small";
      watchBtn.textContent = "Watch";
      watchBtn.addEventListener("click", () => {
        resetSections();
        if (job.status === "done") {
          // play the result on its own, without a before/after comparison
          showCompare(job, false);
          return;
        }
        setBusy(true);
        progressSection.hidden = false;
        pollJob(job.id);
      });
      actions.appendChild(watchBtn);
    }
    if (REMOVABLE_STATUSES.has(job.status)) {
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "small secondary";
      removeBtn.textContent = "×";
      removeBtn.title = "Remove this job and delete its files";
      removeBtn.setAttribute("aria-label", `Remove ${job.filename}`);
      removeBtn.addEventListener("click", async () => {
        if (!confirm(`Remove "${job.filename}" and delete its files?`)) return;
        removeBtn.disabled = true;
        try {
          await fetchJson(`/api/jobs/${job.id}`, { method: "DELETE" });
          forgetSource(job.id);
          loadJobs();
        } catch (err) {
          showError(err.message || String(err));
          removeBtn.disabled = false;
        }
      });
      actions.appendChild(removeBtn);
    }
    jobsBody.appendChild(tr);
  }
}

// -- init ----------------------------------------------------------------

(async () => {
  try {
    const caps = await fetchJson("/api/capabilities");
    pauseResumeSupported = !!caps.pause_resume;
  } catch {
    // Older server process: keep the pause buttons hidden until it restarts.
    pauseResumeSupported = false;
  }
  await Promise.all([loadEngines(), loadGpus(), loadEncoders(), loadInterpolators()]);
  await loadPresets();
  loadJobs();
  // background episodes keep running while the form moves on; keep the list
  // fresh so they can be watched or downloaded without a page reload
  setInterval(loadJobs, 5000);
})();
