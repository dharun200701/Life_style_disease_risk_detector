document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        const predictionResult = document.getElementById('prediction-result');
        predictionResult.textContent = 'Predicted Sleep Disorder: ' + result.prediction;
    })
    .catch(error => {
        console.error('Error:', error);
        const predictionResult = document.getElementById('prediction-result');
        predictionResult.textContent = 'An error occurred. Please try again.';
    });
});