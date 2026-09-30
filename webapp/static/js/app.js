(() => {
  const form = document.querySelector('#detect-form');
  const typeChoices = [...document.querySelectorAll('input[name="media-type"]')];
  const fileInput = document.querySelector('#file-input');
  const dropZone = document.querySelector('#drop-zone');
  const fileMeta = document.querySelector('#file-meta');
  const formatHint = document.querySelector('#format-hint');
  const frameControl = document.querySelector('#frame-control');
  const button = document.querySelector('#detect-button');
  const loading = document.querySelector('#loading-message');
  const previewWrap = document.querySelector('#preview-wrap');
  const imagePreview = document.querySelector('#image-preview');
  const videoPreview = document.querySelector('#video-preview');
  const resultSection = document.querySelector('#result-section');
  const badge = document.querySelector('#prediction-badge');
  const recommendation = document.querySelector('#recommendation');
  const metrics = document.querySelector('#metrics');
  const videoAnalysis = document.querySelector('#video-analysis');
  const gallery = document.querySelector('#suspicious-frames');
  const canvas = document.querySelector('#probability-chart');
  let previewUrl = null;

  const mediaType = () => document.querySelector('input[name="media-type"]:checked').value;
  const percent = value => value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
  const seconds = value => `${Number(value).toFixed(2)} s`;
  const readableSize = size => size < 1024 * 1024 ? `${Math.ceil(size / 1024)} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`;

  function resetPreview() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; previewWrap.hidden = true; imagePreview.hidden = true; videoPreview.hidden = true;
    imagePreview.removeAttribute('src'); videoPreview.removeAttribute('src');
  }

  function setMediaType() {
    const video = mediaType() === 'video';
    fileInput.accept = video ? '.mp4,.avi,.mov' : '.jpg,.jpeg,.png';
    formatHint.textContent = video ? '支持 MP4、AVI、MOV，最大 100 MB' : '支持 JPG、JPEG、PNG，最大 100 MB';
    frameControl.hidden = !video;
    fileInput.value = ''; button.disabled = true; fileMeta.textContent = '尚未选择文件'; resultSection.hidden = true; resetPreview();
  }

  function showFile(file) {
    if (!file) return;
    const image = mediaType() === 'image';
    const allowed = image ? ['jpg', 'jpeg', 'png'] : ['mp4', 'avi', 'mov'];
    const extension = file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(extension)) { fileInput.value = ''; button.disabled = true; fileMeta.textContent = '文件类型与当前模式不匹配'; return; }
    if (file.size > 100 * 1024 * 1024) { fileInput.value = ''; button.disabled = true; fileMeta.textContent = '文件超过 100 MB 限制'; return; }
    resetPreview(); previewUrl = URL.createObjectURL(file); previewWrap.hidden = false;
    if (image) { imagePreview.src = previewUrl; imagePreview.hidden = false; } else { videoPreview.src = previewUrl; videoPreview.hidden = false; }
    fileMeta.textContent = `${file.name} · ${readableSize(file.size)}`; button.disabled = false;
  }

  function metric(label, value, probability = null) {
    const card = document.createElement('div'); card.className = 'metric';
    const labelNode = document.createElement('span'); labelNode.className = 'metric-label'; labelNode.textContent = label;
    const valueNode = document.createElement('strong'); valueNode.className = 'metric-value'; valueNode.textContent = value;
    card.append(labelNode, valueNode);
    if (probability != null) { const bar = document.createElement('div'); bar.className = 'probability'; const fill = document.createElement('i'); fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`; bar.append(fill); card.append(bar); }
    metrics.append(card);
  }

  function setBadge(prediction) {
    badge.textContent = prediction;
    badge.className = `status ${prediction === 'DEEPFAKE' ? 'deepfake' : prediction === 'REAL' ? 'real' : 'uncertain'}`;
  }

  function drawChart(result) {
    const series = [];
    if (result.xception_frame_probabilities) series.push({ name: 'Xception', color: '#2868aa', values: result.xception_frame_probabilities });
    if (result.effort_frame_probabilities) series.push({ name: 'Effort', color: '#bd5a36', values: result.effort_frame_probabilities });
    const context = canvas.getContext('2d'); const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr)); canvas.height = Math.max(1, Math.floor(rect.height * dpr)); context.scale(dpr, dpr);
    const width = rect.width; const height = rect.height; const left = 42; const right = 12; const top = 22; const bottom = 32;
    const x = index => series[0].values.length < 2 ? left : left + index * (width - left - right) / (series[0].values.length - 1);
    const y = value => top + (1 - value) * (height - top - bottom);
    context.clearRect(0, 0, width, height); context.font = '12px Segoe UI'; context.fillStyle = '#657286'; context.strokeStyle = '#dce3eb';
    [0, .5, 1].forEach(value => { context.beginPath(); context.moveTo(left, y(value)); context.lineTo(width - right, y(value)); context.stroke(); context.fillText(value.toFixed(1), 8, y(value) + 4); });
    context.setLineDash([5, 4]); context.strokeStyle = '#a15c00'; context.beginPath(); context.moveTo(left, y(.5)); context.lineTo(width - right, y(.5)); context.stroke(); context.setLineDash([]); context.fillStyle = '#a15c00'; context.fillText('threshold 0.5', width - 92, y(.5) - 5);
    series.forEach((line, lineIndex) => { context.strokeStyle = line.color; context.lineWidth = 2; context.beginPath(); line.values.forEach((value, index) => index ? context.lineTo(x(index), y(value)) : context.moveTo(x(index), y(value))); context.stroke(); context.fillStyle = line.color; context.fillRect(left + lineIndex * 92, 5, 10, 3); context.fillText(line.name, left + 14 + lineIndex * 92, 10); });
    context.fillStyle = '#657286'; context.fillText('Video Time (s)', Math.max(left, width / 2 - 38), height - 7);
  }

  function showResult(result, isVideo) {
    resultSection.hidden = false; metrics.replaceChildren(); setBadge(result.prediction);
    recommendation.hidden = !result.recommendation; recommendation.textContent = result.recommendation === 'Manual review recommended' ? '建议人工复核（Manual Review Recommended）' : result.recommendation || '';
    metric('Prediction', result.prediction); metric('Fake Probability', percent(result.fake_probability), result.fake_probability);
    if (result.xception_probability != null) metric('Xception Fake Probability', percent(result.xception_probability), result.xception_probability);
    if (result.effort_probability != null) metric('Effort Fake Probability', percent(result.effort_probability), result.effort_probability);
    if (result.consensus) metric('Consensus', result.consensus); metric('Inference Time', seconds(result.inference_time));
    videoAnalysis.hidden = !isVideo;
    if (!isVideo) return;
    metric('Total Video Frames', result.total_frames); metric('Sampled Frames', result.sampled_frames); metric('Valid Face Frames', result.valid_faces);
    drawChart(result); gallery.replaceChildren();
    result.suspicious_frames.forEach(frame => { const card = document.createElement('article'); card.className = 'frame-card'; const image = document.createElement('img'); image.src = frame.image; image.alt = '可疑人脸帧'; const caption = document.createElement('p'); caption.textContent = `${Number(frame.timestamp).toFixed(2)} s · ${percent(frame.probability)}`; card.append(image, caption); gallery.append(card); });
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  typeChoices.forEach(choice => choice.addEventListener('change', setMediaType));
  fileInput.addEventListener('change', () => showFile(fileInput.files[0]));
  ['dragenter', 'dragover'].forEach(event => dropZone.addEventListener(event, eventData => { eventData.preventDefault(); dropZone.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach(event => dropZone.addEventListener(event, eventData => { eventData.preventDefault(); dropZone.classList.remove('dragover'); }));
  dropZone.addEventListener('drop', event => { const [file] = event.dataTransfer.files; if (!file) return; const transfer = new DataTransfer(); transfer.items.add(file); fileInput.files = transfer.files; showFile(file); });
  window.addEventListener('resize', () => { if (!videoAnalysis.hidden) drawChart(window.lastVideoResult); });

  form.addEventListener('submit', async event => {
    event.preventDefault(); const file = fileInput.files[0]; if (!file || button.disabled) return;
    const video = mediaType() === 'video'; const data = new FormData(); data.append('file', file); data.append('model', document.querySelector('#model-select').value); if (video) data.append('frames', document.querySelector('#frame-select').value);
    button.disabled = true; loading.hidden = false; resultSection.hidden = true;
    try {
      const response = await fetch(video ? '/api/detect/video' : '/api/detect/image', { method: 'POST', body: data }); const result = await response.json();
      if (!response.ok || !result.success) throw new Error(result.error || '检测请求失败。'); if (video) window.lastVideoResult = result; showResult(result, video);
    } catch (error) { resultSection.hidden = false; metrics.replaceChildren(); recommendation.hidden = false; recommendation.textContent = error.message || '检测失败，请稍后重试。'; badge.textContent = 'ERROR'; badge.className = 'status uncertain'; videoAnalysis.hidden = true; }
    finally { button.disabled = false; loading.hidden = true; }
  });
})();
