import React, { useState, useEffect } from 'react';
import '../styles/analytics.css';

function LiveAnalytics() {
  const [stats, setStats] = useState({
    totalPredictions: 0,
    averageConfidence: 0,
    shotDistribution: {},
    recentPredictions: []
  });

  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    const savedPredictions = localStorage.getItem('predictions');
    if (savedPredictions) {
      try {
        const parsed = JSON.parse(savedPredictions);
        setPredictions(parsed);
        updateStats(parsed);
      } catch (e) {
        console.error('Error loading predictions:', e);
      }
    }
  }, []);

  const updateStats = (predictionList) => {
    if (predictionList.length === 0) return;

    const shotCounts = {};
    let totalConfidence = 0;

    predictionList.forEach(pred => {
      shotCounts[pred.shot] = (shotCounts[pred.shot] || 0) + 1;
      totalConfidence += pred.confidence;
    });

    setStats({
      totalPredictions: predictionList.length,
      averageConfidence: totalConfidence / predictionList.length,
      shotDistribution: shotCounts,
      recentPredictions: predictionList.slice(-5)
    });
  };

  const handleClearHistory = () => {
    if (window.confirm('Clear prediction history?')) {
      setPredictions([]);
      localStorage.removeItem('predictions');
      setStats({
        totalPredictions: 0,
        averageConfidence: 0,
        shotDistribution: {},
        recentPredictions: []
      });
    }
  };

  return (
    <div className="card">
      <h2>Analytics Dashboard</h2>

      <div className="stats-grid">
        <div className="stat-box">
          <h3>Total Predictions</h3>
          <p className="stat-value">{stats.totalPredictions}</p>
        </div>
        <div className="stat-box">
          <h3>Average Confidence</h3>
          <p className="stat-value">{(stats.averageConfidence * 100).toFixed(1)}%</p>
        </div>
      </div>

      <div className="section">
        <h3>Shot Distribution</h3>
        {Object.keys(stats.shotDistribution).length > 0 ? (
          <div className="distribution-chart">
            {Object.entries(stats.shotDistribution).map(([shot, count]) => (
              <div key={shot} className="distribution-item">
                <span className="shot-name">{shot}</span>
                <div className="distribution-bar">
                  <div
                    className="distribution-fill"
                    style={{
                      width: `${(count / Math.max(...Object.values(stats.shotDistribution))) * 100}%`
                    }}
                  ></div>
                </div>
                <span className="shot-count">{count}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="empty-message">No predictions yet</p>
        )}
      </div>

      <div className="section">
        <h3>Recent Predictions</h3>
        {stats.recentPredictions.length > 0 ? (
          <ul className="recent-list">
            {stats.recentPredictions.map((pred, idx) => (
              <li key={idx} className="recent-item">
                <span className="pred-shot">{pred.shot}</span>
                <span className="pred-confidence">{(pred.confidence * 100).toFixed(1)}%</span>
                <span className="pred-time">{new Date(pred.timestamp).toLocaleTimeString()}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty-message">No recent predictions</p>
        )}
      </div>

      <button className="button button-secondary" onClick={handleClearHistory}>
        Clear History
      </button>
    </div>
  );
}

export default LiveAnalytics;
