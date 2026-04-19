import React, { useState, useEffect } from 'react';
import './App.css';
import ImagePredictor from './components/ImagePredictor';
import VideoPredictor from './components/VideoPredictor';
import LiveAnalytics from './components/LiveAnalytics';
import ModelInfo from './components/ModelInfo';
import { API_URL } from './config';

function App() {
  const [activeTab, setActiveTab] = useState('image');
  const [apiStatus, setApiStatus] = useState('checking');
  const [shotClasses, setShotClasses] = useState({});

  useEffect(() => {
    checkApiStatus();
    fetchShotClasses();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (response.ok) {
        setApiStatus('online');
      } else {
        setApiStatus('offline');
      }
    } catch (error) {
      setApiStatus('offline');
      console.error('API health check failed:', error);
    }
  };

  const fetchShotClasses = async () => {
    try {
      const response = await fetch(`${API_URL}/classes`);
      const data = await response.json();
      setShotClasses(data.shot_classes);
    } catch (error) {
      console.error('Failed to fetch shot classes:', error);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <div className="container">
          <h1>Cricket Shot Pose Classifier</h1>
          <p>Real-time pose detection and shot classification system</p>
          <div className="api-status">
            <span className={`status-indicator ${apiStatus}`}></span>
            <span>API: {apiStatus === 'online' ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </header>

      <main className="container">
        <div className="tabs">
          <button
            className={`tab-button ${activeTab === 'image' ? 'active' : ''}`}
            onClick={() => setActiveTab('image')}
          >
            Image Analysis
          </button>
          <button
            className={`tab-button ${activeTab === 'video' ? 'active' : ''}`}
            onClick={() => setActiveTab('video')}
          >
            Video Analysis
          </button>
          <button
            className={`tab-button ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            Analytics
          </button>
          <button
            className={`tab-button ${activeTab === 'info' ? 'active' : ''}`}
            onClick={() => setActiveTab('info')}
          >
            Model Info
          </button>
        </div>

        {activeTab === 'image' && <ImagePredictor shotClasses={shotClasses} />}
        {activeTab === 'video' && <VideoPredictor shotClasses={shotClasses} />}
        {activeTab === 'analytics' && <LiveAnalytics />}
        {activeTab === 'info' && <ModelInfo />}
      </main>

      <footer className="footer">
        <div className="container">
          <p>AMD Developer Hackathon - Cricket Pose Classification</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
