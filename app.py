from flask import Flask, request, jsonify, render_template
import os
import requests
import joblib
import numpy as np
import time
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CACHE_FILE = os.path.join(os.path.dirname(__file__), 'data', 'aqi_cache.json')
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = None
        print(f"Warning: Model not found at {model_path}. Prediction will use mock logic.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# --- API KEY INITIALIZATION ---
# 1. First, check for standard individual keys (The professional way)
openweathermap_api_key = os.getenv("OPENWEATHER_API_KEY", "")
news_api_key = os.getenv("NEWS_API_KEY", "")

# 2. If missing, look for a combined string (comma-separated)
# We check both "API_KEYS" and the accidental "AQI_API_KEY" name seen in Render
combined_keys = os.getenv("API_KEYS") or os.getenv("AQI_API_KEY")

if combined_keys and (not openweathermap_api_key or not news_api_key):
    # Split by comma and clean up whitespace
    keys = [k.strip() for k in combined_keys.split(",")]
    if len(keys) >= 1 and not openweathermap_api_key:
        openweathermap_api_key = keys[0]
    if len(keys) >= 2 and not news_api_key:
        news_api_key = keys[1]

# Log for the user (visible in Render logs)
print(f"Server Initialized | Weather API: {'OK' if openweathermap_api_key else 'MISSING'} | News API: {'OK' if news_api_key else 'MISSING'}")

REAL_COORDS = {
    'Delhi': (28.7041, 77.1025), 'Mumbai': (19.0760, 72.8777), 'Bangalore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707), 'Kolkata': (22.5726, 88.3639), 'Hyderabad': (17.3850, 78.4867),
    'Pune': (18.5204, 73.8567), 'Ahmedabad': (23.0225, 72.5714), 'Jaipur': (26.9124, 75.7873),
    'Lucknow': (26.8467, 80.9462), 'Chandigarh': (30.7333, 76.7794), 'Bhopal': (23.2599, 77.4126),
    'Patna': (25.5941, 85.1376), 'Noida': (28.5355, 77.3910), 'Gurgaon': (28.4595, 77.0266),
    'Kanpur': (26.4499, 80.3319), 'Ghaziabad': (28.6692, 77.4538), 'Agra': (27.1767, 78.0081),
    'Varanasi': (25.3176, 82.9739), 'New Delhi': (28.6139, 77.2090), 'Dwarka': (28.5921, 77.0460),
    'Rohini': (28.7366, 77.1132), 'Faridabad': (28.4089, 77.3178), 'Panipat': (29.3909, 76.9708),
    'Ludhiana': (30.9010, 75.8573), 'Amritsar': (31.6340, 74.8723), 'Udaipur': (24.5854, 73.7125),
    'Jodhpur': (26.2389, 73.0243), 'Kota': (25.2138, 75.8648), 'Nagpur': (21.1458, 79.0882),
    'Nashik': (19.9975, 73.7898), 'Surat': (21.1702, 72.8311), 'Vadodara': (22.3072, 73.1812),
    'Rajkot': (22.3039, 70.8022), 'Panaji': (15.4909, 73.8278), 'Margao': (15.2736, 73.9580),
    'Mysore': (12.2958, 76.6394), 'Hubli': (15.3647, 75.1240), 'Coimbatore': (11.0168, 76.9558),
    'Madurai': (9.9252, 78.1198), 'Warangal': (17.9689, 79.5941), 'Visakhapatnam': (17.6868, 83.2185),
    'Vijayawada': (16.5062, 80.6480), 'Kochi': (9.9312, 76.2673), 'Thiruvananthapuram': (8.5241, 76.9366),
    'Howrah': (22.5958, 88.2636), 'Gaya': (24.7914, 85.0002), 'Bhubaneswar': (20.2961, 85.8245),
    'Cuttack': (20.4625, 85.8828), 'Ranchi': (23.3441, 85.3096), 'Jamshedpur': (22.8046, 86.2029),
    'Indore': (22.7196, 75.8577), 'Gwalior': (26.2124, 78.1772), 'Raipur': (21.2514, 81.6296),
    'Bhilai': (21.1938, 81.3509), 'Leh': (34.1526, 77.5771), 'Srinagar': (34.0837, 74.7973), 
    'Port Blair': (11.6234, 92.7265), 'Shimla': (31.1048, 77.1734), 'Amaravati': (16.5062, 80.6480),
    'Itanagar': (27.0844, 93.6053), 'Imphal': (24.8170, 93.9368), 'Shillong': (25.5788, 91.8933), 
    'Aizawl': (23.7271, 92.7176), 'Kohima': (25.6751, 94.1086), 'Gangtok': (27.3389, 88.6065), 
    'Agartala': (23.8315, 91.2868), 'Dehradun': (30.3165, 78.0322), 'Puducherry': (11.9416, 79.8083)
}

# --- MOCK DATA FALLBACK ---
def get_mock_city_data(city):
    import random
    # Remove the seed so mock data changes every hour (when cache expires)
    # np.random.seed(hash(city) % (2**32)) 
    return {
        'list': [{
            'components': {
                'pm2_5': random.uniform(15, 60),  # Much more normal range (15-60)
                'pm10': random.uniform(30, 100),  # Much more normal range (30-100)
                'no2': random.uniform(10, 40),
                'so2': random.uniform(5, 20),
                'co': random.uniform(400, 800) / 1000.0, 
                'o3': random.uniform(20, 60)
            }
        }]
    }

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/future')
def future():
    return render_template('future.html')

@app.route('/health')
def health():
    return render_template('health.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/news')
def news():
    return render_template('news.html')

# --- API ENDPOINTS ---

# --- PERSISTENT CACHE HELPERS ---

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}

def save_cache(cache_data):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

pollution_cache = load_cache()

def fetch_pollution_data(city):
    now = time.time()
    # Cache for 1 hour (3600 seconds)
    if city in pollution_cache and (now - pollution_cache[city]['timestamp'] < 3600):
        data = pollution_cache[city]['data']
        
        # IMPROVEMENT: If the cache has "mock" data but we now have an API key, 
        # force a fresh fetch to try and get "live" data.
        if data.get('source') in ['mock', 'mock_error'] and openweathermap_api_key:
            pass # Continue to fetch logic below
        else:
            if 'source' not in data:
                data['source'] = 'live' if openweathermap_api_key else 'mock'
            return data

    if not openweathermap_api_key:
        data = get_mock_city_data(city)
        lat, lon = REAL_COORDS.get(city, (20.0, 78.0))
        res = {'city': city, 'lat': lat, 'lon': lon, 'data': data['list'][0]['components'], 'source': 'mock'}
    else:
        try:
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},IN&limit=1&appid={openweathermap_api_key}"
            geo_response = requests.get(geo_url, timeout=5)
            geo_data = geo_response.json()
            if not geo_data: raise ValueError("City not found")
            lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
            ap_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={openweathermap_api_key}"
            ap_response = requests.get(ap_url, timeout=5)
            # Add debug printing to see the real error from OpenWeather
            if ap_response.status_code != 200:
                print(f"OpenWeather Error for {city}: {ap_response.status_code} - {ap_response.text}")
            
            ap_data = ap_response.json()
            res = {'city': city, 'lat': lat, 'lon': lon, 'data': ap_data['list'][0]['components'], 'source': 'live'}
        except Exception as e:
            print(f"Critical Error for {city}: {str(e)}")
            data = get_mock_city_data(city)
            f_lat, f_lon = REAL_COORDS.get(city, (20.0, 78.0))
            res = {'city': city, 'lat': f_lat, 'lon': f_lon, 'data': data['list'][0]['components'], 'source': 'mock_error'}
    
    pollution_cache[city] = {'timestamp': now, 'data': res}
    save_cache(pollution_cache)
    return res

@app.route('/api/pollution')
def api_pollution():
    city = request.args.get('city', 'Delhi')
    return jsonify(fetch_pollution_data(city))

@app.route('/api/pollution_bulk')
def api_pollution_bulk():
    cities_str = request.args.get('cities', 'Delhi,Mumbai')
    cities = [c.strip() for c in cities_str.split(',') if c.strip()]
    
    # Use ThreadPoolExecutor to fetch all cities in parallel
    with ThreadPoolExecutor(max_workers=min(20, len(cities) + 1)) as executor:
        results = list(executor.map(fetch_pollution_data, cities))
        
    if results:
        print(f"📡 API Status Check | City: {results[0]['city']} | Source: {results[0].get('source')} | Key OK: {bool(openweathermap_api_key)}")
        
    return jsonify(results)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    try:
        pm25 = float(data.get('pm2_5', 0))
        pm10 = float(data.get('pm10', 0))
        no2 = float(data.get('no2', 0))
        so2 = float(data.get('so2', 0))
        co = float(data.get('co', 0))
        o3 = float(data.get('o3', 0))
        
        if model:
            features = np.array([[pm25, pm10, no2, so2, co, o3]])
            prediction = model.predict(features)[0]
        else:
            # Fallback mock prediction if model is missing
            def calc_pm25(x): return x*50/30 if x<=30 else 50+(x-30)*50/30 if x<=60 else 100+(x-60)*100/30 if x<=90 else 200+(x-90)*100/30 if x<=120 else 300+(x-120)*100/130 if x<=250 else 400+(x-250)*100/130
            def calc_pm10(x): return x if x<=100 else 100+(x-100)*100/150 if x<=250 else 200+(x-250)*100/100 if x<=350 else 300+(x-350)*100/80 if x<=430 else 400+(x-430)*100/80
            def calc_no2(x): return x*50/40 if x<=40 else 50+(x-40)*50/40 if x<=80 else 100+(x-80)*100/100 if x<=180 else 200+(x-180)*100/100 if x<=280 else 300+(x-280)*100/120 if x<=400 else 400+(x-400)*100/120
            def calc_so2(x): return x*50/40 if x<=40 else 50+(x-40)*50/40 if x<=80 else 100+(x-80)*100/300 if x<=380 else 200+(x-380)*100/420 if x<=800 else 300+(x-800)*100/800 if x<=1600 else 400+(x-1600)*100/800
            def calc_co(x): return x*50/1 if x<=1 else 50+(x-1)*50/1 if x<=2 else 100+(x-2)*100/8 if x<=10 else 200+(x-10)*100/7 if x<=17 else 300+(x-17)*100/17 if x<=34 else 400+(x-34)*100/17
            def calc_o3(x): return x if x<=50 else 50+(x-50)*50/50 if x<=100 else 100+(x-100)*100/68 if x<=168 else 200+(x-168)*100/40 if x<=208 else 300+(x-208)*100/540 if x<=748 else 400+(x-748)*100/540

            prediction = max(
                calc_pm25(pm25), calc_pm10(pm10), calc_no2(no2), 
                calc_so2(so2), calc_co(co/1000.0), calc_o3(o3)
            )
            
        prediction = max(10, min(500, int(prediction))) # cap it
        
        return jsonify({'predicted_aqi': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict_future')
def api_predict_future():
    city = request.args.get('city')
    # Fetch current data first
    response = api_pollution()
    if isinstance(response, tuple):
        return response # Handle error tuple
    
    res_json = response.get_json()
    if 'data' not in res_json:
        return jsonify({'error': 'Failed to fetch current data'}), 400
        
    current_comps = res_json['data']
    
    # Predict tomorrow (add some random noise)
    import random
    future_comps = {}
    for k, v in current_comps.items():
        if k in ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']:
            future_comps[k] = v * random.uniform(0.8, 1.3)
            
    try:
        if model:
            # Current
            curr_f = np.array([[current_comps.get('pm2_5',0), current_comps.get('pm10',0), current_comps.get('no2',0), 
                                current_comps.get('so2',0), current_comps.get('co',0), current_comps.get('o3',0)]])
            curr_aqi = model.predict(curr_f)[0]
            
            # Future
            fut_f = np.array([[future_comps.get('pm2_5',0), future_comps.get('pm10',0), future_comps.get('no2',0), 
                                future_comps.get('so2',0), future_comps.get('co',0), future_comps.get('o3',0)]])
            fut_aqi = model.predict(fut_f)[0]
        else:
            def calc_mock(comps):
                pm25 = comps.get('pm2_5',0)
                pm10 = comps.get('pm10',0)
                no2 = comps.get('no2',0)
                so2 = comps.get('so2',0)
                co = comps.get('co',0)/1000.0
                o3 = comps.get('o3',0)

                def cpm25(x): return x*50/30 if x<=30 else 50+(x-30)*50/30 if x<=60 else 100+(x-60)*100/30 if x<=90 else 200+(x-90)*100/30 if x<=120 else 300+(x-120)*100/130 if x<=250 else 400+(x-250)*100/130
                def cpm10(x): return x if x<=100 else 100+(x-100)*100/150 if x<=250 else 200+(x-250)*100/100 if x<=350 else 300+(x-350)*100/80 if x<=430 else 400+(x-430)*100/80
                def cno2(x): return x*50/40 if x<=40 else 50+(x-40)*50/40 if x<=80 else 100+(x-80)*100/100 if x<=180 else 200+(x-180)*100/100 if x<=280 else 300+(x-280)*100/120 if x<=400 else 400+(x-400)*100/120
                def cso2(x): return x*50/40 if x<=40 else 50+(x-40)*50/40 if x<=80 else 100+(x-80)*100/300 if x<=380 else 200+(x-380)*100/420 if x<=800 else 300+(x-800)*100/800 if x<=1600 else 400+(x-1600)*100/800
                def cco(x): return x*50/1 if x<=1 else 50+(x-1)*50/1 if x<=2 else 100+(x-2)*100/8 if x<=10 else 200+(x-10)*100/7 if x<=17 else 300+(x-17)*100/17 if x<=34 else 400+(x-34)*100/17
                def co3(x): return x if x<=50 else 50+(x-50)*50/50 if x<=100 else 100+(x-100)*100/68 if x<=168 else 200+(x-168)*100/40 if x<=208 else 300+(x-208)*100/540 if x<=748 else 400+(x-748)*100/540
                
                return max(cpm25(pm25), cpm10(pm10), cno2(no2), cso2(so2), cco(co), co3(o3))
            curr_aqi = calc_mock(current_comps)
            fut_aqi = calc_mock(future_comps)
            
        curr_aqi = max(10, min(500, int(curr_aqi)))
        fut_aqi = max(10, min(500, int(fut_aqi)))
        
        return jsonify({
            'city': city,
            'current_aqi': curr_aqi,
            'future_aqi': fut_aqi,
            'current_components': current_comps,
            'future_components': future_comps
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/news')
def api_news():
    import random
    from datetime import datetime, timedelta
    import json
    
    # Load the news API key (may be empty if not configured)
    
    
    # Helper to generate mock articles (same as previous implementation)
    def generate_mock_articles():
        import urllib.parse
        cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Beijing', 'Los Angeles', 'London']
        issues = ['Severe Smog', 'PM2.5 Spikes', 'Air Quality Emergency', 'Vehicle Emissions Reduction', 'Industrial Pollution', 'Stubble Burning', 'Renewable Energy Initiative']
        articles = []
        for i in range(6):
            city = random.choice(cities)
            issue = random.choice(issues)
            severity = "high" if "Emergency" in issue or "Severe" in issue or "Spike" in issue else "medium"
            tag_color = "#ef4444" if severity == "high" else "#eab308"
            if "Initiative" in issue or "Reduction" in issue:
                tag_color = "#10b981"
                severity = "low"
            # Reduced offset to 1-6 hours so mock news feels "fresh" and from Today
            time_offset = random.randint(1, 6)
            time_str = (datetime.now() - timedelta(hours=time_offset)).strftime("%b %d, %Y - %I:%M %p")
            headline = f"Alert: {city} Reports {issue} Approaching Record Levels"
            if severity == "low":
                headline = f"{city} Implements New {issue} to Combat Air Quality Crisis"
            # Build a Google search URL so the mock link leads to a useful page
            query = urllib.parse.quote(headline)
            search_url = f"https://www.google.com/search?q={query}"
            articles.append({
                "id": i,
                "headline": headline,
                "source": random.choice(["Global Environment Watch", "Climate News Daily", "AirQuality Network", "Earth Monitor"]),
                "timestamp": time_str,
                "summary": f"Local authorities in {city} are heavily monitoring the recent developments regarding {issue.lower()}. Experts warn that immediate policy interventions may be required as meteorological conditions continue to trap airborne particulate matter near the surface level.",
                "severity_color": tag_color,
                "url": search_url
            })
        return articles
    
    # If we have a valid NEWS_API_KEY, try to fetch real news
    if news_api_key:
        try:
            # Query focused on air quality and pollution topics
            query = "air quality pollution smog"
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&language=en&sortBy=publishedAt&pageSize=6&apiKey={news_api_key}"
            )
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Transform NewsAPI response to the shape expected by the front‑end
            articles = []
            for i, article in enumerate(data.get('articles', [])):
                title = article.get('title', 'No title')
                source_name = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', datetime.utcnow().isoformat())
                # Convert ISO timestamp to our display format
                try:
                    dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    timestamp = dt.strftime("%b %d, %Y - %I:%M %p")
                except Exception:
                    timestamp = published
                summary = article.get('description') or article.get('content') or ''
                url = article.get('url', '#')
                # Simple severity heuristic based on keywords in the title
                lowered = title.lower()
                if any(word in lowered for word in ['severe', 'alert', 'spike', 'emergency']):
                    severity_color = "#ef4444"
                elif any(word in lowered for word in ['initiative', 'policy', 'reduction']):
                    severity_color = "#10b981"
                else:
                    severity_color = "#eab308"
                articles.append({
                    "id": i,
                    "headline": title,
                    "source": source_name,
                    "timestamp": timestamp,
                    "summary": summary,
                    "severity_color": severity_color,
                    "url": url
                })
            # If the external API returned no articles, fall back to mock
            if not articles:
                articles = generate_mock_articles()
            return jsonify({"status": "success", "articles": articles})
        except Exception as e:
            # Log the error server‑side and fall back to mock data for resilience
            print(f"News API error: {e}")
            articles = generate_mock_articles()
            return jsonify({"status": "success", "articles": articles})
    else:
        # No API key – use mock data directly
        articles = generate_mock_articles()
        return jsonify({"status": "success", "articles": articles})

if __name__ == '__main__':
    # Use dynamic port for Render/Heroku or default to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
