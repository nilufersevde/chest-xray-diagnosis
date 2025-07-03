const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    alert('Please select a file.');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  resultDiv.textContent = 'Predicting...';

  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Server error');
    }

    const data = await response.json();
    resultDiv.innerHTML = `
      <strong>Prediction:</strong> ${data.prediction}<br/>
      <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%<br/>
      <strong>Probabilities:</strong><br/>
      NORMAL: ${(data.probabilities.NORMAL * 100).toFixed(2)}%<br/>
      PNEUMONIA: ${(data.probabilities.PNEUMONIA * 100).toFixed(2)}%
    `;
  } catch (err) {
    console.error(err);
    resultDiv.textContent = 'Prediction failed.';
  }
});
