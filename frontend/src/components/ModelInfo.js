import React, { useState, useEffect } from 'react';
import '../styles/model-info.css';

function ModelInfo() {
  const [modelInfo, setModelInfo] = useState(null);
  const [classes, setClasses] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelInfo();
    fetchClasses();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await fetch('http://localhost:8000/model/info');
      const data = await response.json();
      setModelInfo(data);
    } catch (err) {
      setError('Failed to fetch model information: ' + err.message);
    }
  };

  const fetchClasses = async () => {
    try {
      const response = await fetch('http://localhost:8000/classes');
      const data = await response.json();
      setClasses(data);
    } catch (err) {
      console.error('Failed to fetch classes:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Model Information</h2>

      {error && <div className="error">{error}</div>}

      {loading ? (
        <div style={{ textAlign: 'center' }}>
          <span className="loading"></span>
        </div>
      ) : (
        <>
          {modelInfo && (
            <div className="info-sections">
              <section className="info-section">
                <h3>Configuration</h3>
                <div className="info-item">
                  <label>Device</label>
                  <value>{modelInfo.device}</value>
                </div>
                <div className="info-item">
                  <label>Pose Confidence Threshold</label>
                  <value>{modelInfo.confidence_threshold}</value>
                </div>
                <div className="info-item">
                  <label>Tracking Confidence</label>
                  <value>{modelInfo.tracking_confidence}</value>
                </div>
              </section>

              <section className="info-section">
                <h3>Upload Limits</h3>
                <div className="info-item">
                  <label>Max Upload Size</label>
                  <value>{(modelInfo.max_upload_size / (1024 * 1024)).toFixed(1)} MB</value>
                </div>
                <div className="info-item">
                  <label>Supported Video Formats</label>
                  <value>{modelInfo.supported_video_formats.join(', ')}</value>
                </div>
                <div className="info-item">
                  <label>Supported Image Formats</label>
                  <value>{modelInfo.supported_image_formats.join(', ')}</value>
                </div>
              </section>

              <section className="info-section">
                <h3>Shot Classes ({classes.total_classes})</h3>
                <div className="classes-grid">
                  {classes.shot_classes && Object.entries(classes.shot_classes).map(([id, className]) => (
                    <div key={id} className="class-tag">
                      {className}
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default ModelInfo;
