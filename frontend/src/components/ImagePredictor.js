import React, { useState, useRef } from 'react';
import '../styles/predictor.css';

function ImagePredictor({ shotClasses }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onload = (event) => setPreview(event.target.result);
      reader.readAsDataURL(file);
      setError(null);
      setResult(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onload = (event) => setPreview(event.target.result);
      reader.readAsDataURL(file);
      setError(null);
      setResult(null);
    }
  };

  const handlePredict = async () => {
    if (!image) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('http://localhost:8000/predict/image', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError(data.message || 'Prediction failed');
      }
    } catch (err) {
      setError('Error connecting to API: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="card">
      <h2>Image Analysis</h2>

      <div
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <div className="file-upload">
          <input
            ref={fileInputRef}
            type="file"
            id="image-input"
            onChange={handleFileChange}
            accept="image/*"
          />
          <label htmlFor="image-input" className="file-upload-label">
            Drop image here or click to select
          </label>
        </div>
      </div>

      {preview && (
        <div className="preview-section">
          <img src={preview} alt="Preview" className="image-preview" />
        </div>
      )}

      <div className="action-buttons">
        <button
          className="button"
          onClick={handlePredict}
          disabled={!image || loading}
        >
          {loading ? <span className="loading"></span> : 'Analyze Image'}
        </button>
        {image && (
          <button className="button button-secondary" onClick={handleClear}>
            Clear
          </button>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="results">
          <div className="result-item">
            <h3>Predicted Shot</h3>
            <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#667eea' }}>
              {result.predicted_shot.toUpperCase()}
            </p>
            <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${result.confidence * 100}%` }}
              ></div>
            </div>
          </div>

          {result.angle_features && (
            <div className="angle-features">
              <div className="feature-box">
                <h4>Left Arm Angle</h4>
                <p>{result.angle_features.left_arm_angle.toFixed(1)}°</p>
              </div>
              <div className="feature-box">
                <h4>Right Arm Angle</h4>
                <p>{result.angle_features.right_arm_angle.toFixed(1)}°</p>
              </div>
              <div className="feature-box">
                <h4>Left Leg Angle</h4>
                <p>{result.angle_features.left_leg_angle.toFixed(1)}°</p>
              </div>
              <div className="feature-box">
                <h4>Right Leg Angle</h4>
                <p>{result.angle_features.right_leg_angle.toFixed(1)}°</p>
              </div>
              <div className="feature-box">
                <h4>Shoulder Angle</h4>
                <p>{result.angle_features.shoulder_angle.toFixed(1)}°</p>
              </div>
              <div className="feature-box">
                <h4>Hip Angle</h4>
                <p>{result.angle_features.hip_angle.toFixed(1)}°</p>
              </div>
            </div>
          )}

          <div className="result-item">
            <h3>All Predictions</h3>
            <ul className="predictions-list">
              {Object.entries(result.all_predictions).map(([shot, probability]) => (
                <li key={shot}>
                  <span>{shot}</span>
                  <span>{(probability * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImagePredictor;
