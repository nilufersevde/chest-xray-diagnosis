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
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Server error');
    }

    const data = await response.json();
    
    if (data.prediction === 'UNCERTAIN') {
      resultDiv.innerHTML = `
        <div style="color: #f57c00; background-color: #fff3e0; padding: 10px; border-radius: 5px; margin: 10px 0;">
          <strong>🤔 Uncertain:</strong> The model is not confident enough to make a prediction.<br/>
          <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%<br/>
          <strong>Probabilities:</strong><br/>
          NORMAL: ${(data.probabilities.NORMAL * 100).toFixed(2)}%<br/>
          PNEUMONIA: ${(data.probabilities.PNEUMONIA * 100).toFixed(2)}%
        </div>
      `;
    } else {
      resultDiv.innerHTML = `
        <div style="color: #2e7d32; background-color: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0;">
          <strong>Prediction:</strong> ${data.prediction}<br/>
          <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%<br/>
          <strong>Probabilities:</strong><br/>
          NORMAL: ${(data.probabilities.NORMAL * 100).toFixed(2)}%<br/>
          PNEUMONIA: ${(data.probabilities.PNEUMONIA * 100).toFixed(2)}%
        </div>
      `;
    }
  } catch (err) {
    console.error(err);
    resultDiv.textContent = 'Prediction failed.';
  }
});
