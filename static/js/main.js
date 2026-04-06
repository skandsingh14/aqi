const INDIAN_CITIES = [
    'Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur',
    'Lucknow', 'Chandigarh', 'Bhopal', 'Patna', 'Noida', 'Gurgaon', 'Kanpur', 'Ghaziabad', 'Agra',
    'Varanasi', 'New Delhi', 'Dwarka', 'Rohini', 'Faridabad', 'Panipat', 'Ludhiana', 'Amritsar',
    'Udaipur', 'Jodhpur', 'Kota', 'Nagpur', 'Nashik', 'Surat', 'Vadodara', 'Rajkot', 'Panaji', 'Margao',
    'Mysore', 'Hubli', 'Coimbatore', 'Madurai', 'Warangal', 'Visakhapatnam', 'Vijayawada', 'Kochi',
    'Thiruvananthapuram', 'Howrah', 'Gaya', 'Bhubaneswar', 'Cuttack', 'Ranchi', 'Jamshedpur', 'Indore',
    'Gwalior', 'Raipur', 'Bhilai', 'Leh', 'Srinagar', 'Port Blair', 'Shimla', 'Amaravati', 'Itanagar', 
    'Imphal', 'Shillong', 'Aizawl', 'Kohima', 'Gangtok', 'Agartala', 'Dehradun', 'Puducherry'
];

// Helper to get AQI category and color class based on the Indian AQI rules or general approximation
function getAqiInfo(aqi) {
    if (aqi <= 50) return { label: 'Good', colorClass: 'text-good', bgClass: 'bg-good', colorHex: '#10b981' };
    if (aqi <= 100) return { label: 'Moderate', colorClass: 'text-moderate', bgClass: 'bg-moderate', colorHex: '#eab308' };
    if (aqi <= 200) return { label: 'Poor', colorClass: 'text-poor', bgClass: 'bg-poor', colorHex: '#f97316' };
    if (aqi <= 300) return { label: 'Very Poor', colorClass: 'text-vpoor', bgClass: 'bg-vpoor', colorHex: '#ef4444' };
    return { label: 'Severe', colorClass: 'text-severe', bgClass: 'bg-severe', colorHex: '#9f1239' };
}

// Global helper to update topbar API status
function updateapiStatus(source) {
    const text = document.getElementById('api-status-text');
    const dot = document.getElementById('api-status-dot');
    const wrapper = document.getElementById('api-status-wrapper');
    
    if (!text || !dot) return;
    
    if (source === 'live') {
        text.innerText = 'Live API Data';
        dot.style.backgroundColor = '#10b981';
        wrapper.classList.remove('status-mock');
    } else {
        text.innerText = 'Mock Data (API Offline)';
        dot.style.backgroundColor = '#eab308';
        wrapper.classList.add('status-mock');
    }
}

// Dynamically add staggered animations to all panel cards on load
document.addEventListener("DOMContentLoaded", () => {
    const cards = document.querySelectorAll('.card, .glass-panel');
    cards.forEach((card, index) => {
        card.classList.add('animate-in');
        card.style.animationDelay = `${index * 0.08}s`; // Sped up the cards
    });

    // Mobile Menu Toggle
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const sidebar = document.querySelector('.sidebar');
    
    if (mobileMenuBtn && sidebar) {
        mobileMenuBtn.addEventListener('click', () => {
            sidebar.classList.toggle('active');
            const icon = mobileMenuBtn.querySelector('i');
            if (sidebar.classList.contains('active')) {
                icon.classList.replace('fa-bars-staggered', 'fa-xmark');
            } else {
                icon.classList.replace('fa-xmark', 'fa-bars-staggered');
            }
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 992 && 
            sidebar && sidebar.classList.contains('active') && 
            !sidebar.contains(e.target) && 
            !mobileMenuBtn.contains(e.target)) {
            sidebar.classList.remove('active');
            mobileMenuBtn.querySelector('i').classList.replace('fa-xmark', 'fa-bars-staggered');
        }
    });
});

// Global approximation formula for AQI display matching Indian Standards (Max Sub-Index)
function calculateProxyAqi(comps) {
    if (!comps) return 0;
    
    function calc_pm25(x) {
        if (!x) return 0;
        if (x <= 30) return x * 50 / 30;
        else if (x <= 60) return 50 + (x - 30) * 50 / 30;
        else if (x <= 90) return 100 + (x - 60) * 100 / 30;
        else if (x <= 120) return 200 + (x - 90) * 100 / 30;
        else if (x <= 250) return 300 + (x - 120) * 100 / 130;
        else return 400 + (x - 250) * 100 / 130;
    }

    function calc_pm10(x) {
        if (!x) return 0;
        if (x <= 50) return x;
        else if (x <= 100) return x;
        else if (x <= 250) return 100 + (x - 100) * 100 / 150;
        else if (x <= 350) return 200 + (x - 250) * 100 / 100;
        else if (x <= 430) return 300 + (x - 350) * 100 / 80;
        else return 400 + (x - 430) * 100 / 80;
    }

    function calc_pm10(x) {
        if (!x) return 0;
        if (x <= 50) return x;
        else if (x <= 100) return x;
        else if (x <= 250) return 100 + (x - 100) * 100 / 150;
        else if (x <= 350) return 200 + (x - 250) * 100 / 100;
        else if (x <= 430) return 300 + (x - 350) * 100 / 80;
        else return 400 + (x - 430) * 100 / 80;
    }

    function calc_no2(x) {
        if (!x) return 0;
        if (x <= 40) return x * 50 / 40;
        else if (x <= 80) return 50 + (x - 40) * 50 / 40;
        else if (x <= 180) return 100 + (x - 80) * 100 / 100;
        else if (x <= 280) return 200 + (x - 180) * 100 / 100;
        else if (x <= 400) return 300 + (x - 280) * 100 / 120;
        else return 400 + (x - 400) * 100 / 120;
    }

    function calc_so2(x) {
        if (!x) return 0;
        if (x <= 40) return x * 50 / 40;
        else if (x <= 80) return 50 + (x - 40) * 50 / 40;
        else if (x <= 380) return 100 + (x - 80) * 100 / 300;
        else if (x <= 800) return 200 + (x - 380) * 100 / 420;
        else if (x <= 1600) return 300 + (x - 800) * 100 / 800;
        else return 400 + (x - 1600) * 100 / 800;
    }

    function calc_co(x) { // Note: expecting mg/m3
        if (!x) return 0;
        if (x <= 1.0) return x * 50 / 1.0;
        else if (x <= 2.0) return 50 + (x - 1.0) * 50 / 1.0;
        else if (x <= 10.0) return 100 + (x - 2.0) * 100 / 8.0;
        else if (x <= 17.0) return 200 + (x - 10.0) * 100 / 7.0;
        else if (x <= 34.0) return 300 + (x - 17.0) * 100 / 17.0;
        else return 400 + (x - 34.0) * 100 / 17.0;
    }

    function calc_o3(x) {
        if (!x) return 0;
        if (x <= 50) return x * 50 / 50;
        else if (x <= 100) return 50 + (x - 50) * 50 / 50;
        else if (x <= 168) return 100 + (x - 100) * 100 / 68;
        else if (x <= 208) return 200 + (x - 168) * 100 / 40;
        else if (x <= 748) return 300 + (x - 208) * 100 / 540;
        else return 400 + (x - 748) * 100 / 540;
    }

    // AQI is the highest sub-index of any single pollutant
    let aqi = Math.max(
        calc_pm25(comps.pm2_5),
        calc_pm10(comps.pm10),
        calc_no2(comps.no2),
        calc_so2(comps.so2),
        calc_co((comps.co||0)/1000.0), // OpenWeather CO is ug/m3
        calc_o3(comps.o3)
    );
    return Math.min(500, Math.max(10, Math.floor(aqi)));
}
