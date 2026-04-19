/**
 * API Configuration
 * Change API_URL to your Colab backend URL
 * 
 * Local: http://localhost:8000
 * Colab: https://YOUR_NGROK_URL.ngrok.io
 */

// LOCAL (for local backend)
// export const API_URL = 'http://localhost:8000';

// COLAB (replace with your Colab ngrok URL)
export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
