import React, { useState, useRef } from 'react';
import '../styles/predictor.css';

function VideoPredictor({ shotClasses }) {
  const [video, setVideo] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideo(file);
      const url = URL.createObjectURL(file);
      setPreview(url);
      setError(null);
      setResult(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.includes('video')) {
      setVideo(file);
      const url = URL.createObjectURL(file);
      setPreview(url);
      setError(null);
      setResult(null);
    }
  };

  const handlePredict = async () => {
    if (!video) {
      setError('Please select a video first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', video);

    try {
      const response = await fetch('http://localhost:8000/predict/video', {
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
    setVideo(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="card">
      <h2>Video Analysis</h2>

      <div
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <div className="file-upload">
          <input
            ref={fileInputRef}
            type="file"
            id="video-input"
            onChange={handleFileChange}
            accept="video/*"
          />
          <label htmlFor="video-input" className="file-upload-label">
            Drop video here or click to select
          </label>
        </div>
      </div>

      {preview && (
        <div className="preview-section">
          <video
            src={preview}
            controls
            className="video-preview"
            style={{ width: '100%', maxWidth: '400px', marginTop: '20px' }}
          />
        </div>
      )}

      <div className="action-buttons">
        <button
          className="button"
          onClick={handlePredict}
          disabled={!video || loading}
        >
          {loading ? <span className="loading"></span> : 'Analyze Video'}
        </button>
        {video && (
          <button className="button button-secondary" onClick={handleClear}>
            Clear
          </button>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="results">
          <div className="result-item">
            <h3>Dominant Shot Classification</h3>
            <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#667eea' }}>
              {(result.shot_class || 'UNKNOWN').toUpperCase()}
            </p>
            <p>Average Confidence: {(result.average_confidence * 100 || result.confidence * 100).toFixed(2)}%</p>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${(result.average_confidence || result.confidence) * 100}%` }}
              ></div>
            </div>
          </div>

          <div className="result-item">
            <h3>Analysis Details</h3>
            <p>Frames Processed: {result.frames_processed}</p>
            <p>Total Predictions: {result.predictions_detail?.length || 'N/A'}</p>
          </div>

          <div className="result-item">
            <h3>All Predictions</h3>
            <ul className="predictions-list">
              {(result.probabilities || {}) && Object.entries(result.probabilities || {}).map(([shot, probability]) => (
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

export default VideoPredictor;
