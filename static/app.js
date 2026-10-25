const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const previewImage = document.getElementById('previewImage');
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const startCameraBtn = document.getElementById('startCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const statusText = document.getElementById('status');
const resultBox = document.getElementById('result');
const className = document.getElementById('className');
const confidence = document.getElementById('confidence');
const topClasses = document.getElementById('topClasses');

let stream = null;
let currentImage = null;

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.style.color = isError ? '#b42318' : '#214d3f';
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.style.display = 'block';
    previewImage.style.display = 'none';
    document.querySelector('.preview-box').classList.add('has-video');
    document.querySelector('.preview-box').classList.remove('has-image');
    setStatus('Camera ready. Capture a photo when you are ready.');
  } catch (error) {
    setStatus('Camera access denied or unavailable.', true);
    console.error(error);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
}

function handlePreviewImage(file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    previewImage.src = event.target.result;
    previewImage.style.display = 'block';
    video.style.display = 'none';
    document.querySelector('.preview-box').classList.add('has-image');
    document.querySelector('.preview-box').classList.remove('has-video');
    currentImage = file;
    setStatus('Image loaded. Click “Check deficiency”.');
  };
  reader.readAsDataURL(file);
}

async function captureFrame() {
  if (!stream) {
    setStatus('Start the camera first.', true);
    return;
  }

  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, width, height);

  canvas.toBlob(async (blob) => {
    if (!blob) {
      setStatus('Could not capture image.', true);
      return;
    }

    const file = new File([blob], 'capture.png', { type: 'image/png' });
    handlePreviewImage(file);
  }, 'image/png');
}

async function sendPrediction(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/predict', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Prediction failed');
  }

  return response.json();
}

predictBtn.addEventListener('click', async () => {
  const file = currentImage || fileInput.files[0];
  if (!file) {
    setStatus('Please choose an image or capture a photo first.', true);
    return;
  }

  setStatus('Analyzing image...');
  try {
    const result = await sendPrediction(file);
    resultBox.classList.remove('hidden');
    className.textContent = result.class_name;
    confidence.textContent = result.confidence;

    topClasses.innerHTML = result.top_classes
      .map(
        (item) => `
          <div class="top-item">
            <span>${item.label}</span>
            <span>${item.confidence.toFixed(2)}%</span>
          </div>
        `
      )
      .join('');

    setStatus('Prediction complete.');
  } catch (error) {
    console.error(error);
    setStatus('Could not classify this image. Please use a clear photo of a leaf.', true);
  }
});

fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    currentImage = file;
    handlePreviewImage(file);
  }
});

startCameraBtn.addEventListener('click', startCamera);
captureBtn.addEventListener('click', captureFrame);
window.addEventListener('beforeunload', stopCamera);
