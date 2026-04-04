# 🌍 Air Quality Prediction System (EcoQ India)

Welcome to the **Advanced Air Quality Prediction System**, a premium, data-driven platform designed for high-impact environmental analysis and forecasting.

## ✨ Features
- **Machine Learning AQI Prediction**: A Random Forest model trained on 250,000+ Indian environmental records (CPCB).
- **Premium Glassmorphism UI**: A stunning, modern interface with ambient animations and reactive elements.
- **Live Global Fetching**: Real-time pollution data from OpenWeatherMap and environmental news via NewsAPI.
- **Performance Optimized**: Parallel fetching and persistent caching for near-instant data loading.
- **Interactive Visuals**: Full India Map (Leaflet.js) and Live Analytics (Chart.js).

## 🚀 Getting Started

### 1. Requirements
- Python 3.8+
- [Optional] OpenWeatherMap API Key & NewsAPI Key

### 2. Setup
1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   .\venv\Scripts\activate   # Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Keys**:
   Create a `.env` file in the root directory and add your keys:
   ```dotenv
   OPENWEATHER_API_KEY=your_key_here
   NEWS_API_KEY=your_key_here
   ```

4. **Run the App**:
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

## 📁 Project Structure
- `app.py`: Backend Flask controller & Caching Logic.
- `model/`: Contains `model.pkl` (The trained Random Forest).
- `templates/`: HTML structures with Jinja2.
- `static/`: CSS styling (Glassmorphism) and JavaScript interactivity.
- `data/`: Local storage for the AQI Cache.

---
*Created with 💚 for the future of our environment.*
