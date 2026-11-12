'''DeliveryETA Streamlit UI connected to the FastAPI /predict endpoint.'''

import json
import math
import os
from datetime import date, time
import pandas as pd
import requests
import streamlit as st

from feature_engineering import prepare_prediction_json


API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000/predict')

st.set_page_config(
    page_title='DeliveryETA | Swiggy Delivery Time Prediction',
    page_icon='🛵',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.html('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [data-testid='stApp'] {
    font-family: 'DM Sans', sans-serif;
}
[data-testid='stApp'] {
    background: linear-gradient(115deg, #fffaf6 0%, #fff 52%, #f5f8fc 100%);
}
[data-testid='stHeader'] {
    background: transparent;
}

/* Leave space below Streamlit's header and keep the content centered. */
[data-testid='stMainBlockContainer'],
.block-container {
    max-width: 1320px;
    padding-top: 5rem !important;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
[data-testid='stMain'] {
    overflow-x: clip;
}

/* Refined sidebar: branding, information hierarchy and responsive spacing. */
[data-testid='stSidebar'] {
    background: linear-gradient(180deg, #fffaf5 0%, #fffdfb 62%, #fff5ec 100%);
    border-right: 1px solid #f1e7df;
}
[data-testid='stSidebar'] [data-testid='stSidebarContent'] {
    padding-top: 1rem;
}
.sidebar-brand-card {
    background: #17253e;
    border-radius: 18px;
    padding: 20px 17px 18px;
    margin: 10px 0 20px;
    box-shadow: 0 9px 25px rgba(21, 32, 55, .10);
}
.sidebar-brand-row { display: flex; align-items: center; gap: 12px; }
.sidebar-logo {
    background: rgba(252, 128, 25, .17);
    border: 1px solid rgba(252, 128, 25, .30);
    border-radius: 12px;
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    flex-shrink: 0;
}
.sidebar-brand-name {
    font-size: 22px;
    line-height: 1.15;
    color: #ffffff;
    font-weight: 800;
    letter-spacing: -.7px;
}
.sidebar-brand-name span { color: #ffa24f; }
.sidebar-brand-subtitle {
    color: #cad5e5;
    font-size: 11px;
    line-height: 1.6;
    margin: 13px 0 0;
}
.sidebar-section-title {
    color: #17253e;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .095em;
    margin: 19px 0 9px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-section-title::before {
    content: '';
    background: #fc8019;
    height: 14px;
    width: 3px;
    border-radius: 3px;
}
.sidebar-info-card {
    background: #ffffff;
    border: 1px solid #f0e6dd;
    border-radius: 13px;
    padding: 14px 15px;
    box-shadow: 0 3px 14px rgba(37, 43, 61, .025);
}
.sidebar-info-card p {
    color: #546179;
    font-size: 12.5px;
    line-height: 1.7;
    margin: 0;
}
.sidebar-chip-row { display: flex; flex-wrap: wrap; gap: 7px; }
.sidebar-chip {
    display: inline-block;
    background: #fff2e8;
    border: 1px solid #ffe0c7;
    color: #8f4d17;
    border-radius: 7px;
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 700;
}
.sidebar-note {
    margin-top: 20px;
    border-radius: 12px;
    padding: 13px 14px;
    background: #eef4fa;
    border: 1px solid #dce8f5;
    color: #4a6078;
    font-size: 11.5px;
    line-height: 1.6;
}
.sidebar-footer {
    color: #9a8790;
    text-align: center;
    font-size: 10px;
    margin-top: 22px;
    padding-bottom: 8px;
}



/* Developer contact links. */
.sidebar-contact-card {
    background: #ffffff;
    border: 1px solid #f0e6dd;
    border-radius: 13px;
    padding: 12px 14px;
}
.sidebar-contact-link {
    display: block;
    color: #1f4d79 !important;
    font-size: 12px;
    font-weight: 700;
    text-decoration: none !important;
    overflow-wrap: anywhere;
    padding: 6px 0;
}
.sidebar-contact-link:hover { color: #e86e0d !important; text-decoration: underline !important; }

/* Hero banner: enough padding to prevent the kicker from being clipped. */
.hero {
    position: relative;
    box-sizing: border-box;
    background: linear-gradient(110deg, #14213d, #263d59);
    color: white;
    border-radius: 20px;
    padding: 34px 36px;
    margin-top: 0;
    margin-bottom: 26px;
    overflow: visible;
}
.hero h1 {
    color: white;
    font-size: 2.25rem;
    font-weight: 800;
    line-height: 1.25;
    margin: 8px 0 12px;
}
.hero p {
    color: #dbe5f3;
    font-size: 1rem;
    line-height: 1.65;
    margin: 0;
}
.kicker {
    display: block;
    color: #ffb878;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .11em;
    line-height: 1.5;
    margin: 0 0 8px;
}

/* Form and input spacing. */
[data-testid='stForm'] {
    background: #fff;
    border: 1px solid #e9edf3;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(25, 41, 66, .05);
}
[data-testid='stFormSubmitButton'] button {
    background: #fc8019;
    color: #fff;
    border-radius: 11px;
    border: 0;
    min-height: 48px;
    font-weight: 700;
}
[data-testid='stFormSubmitButton'] button:hover {
    background: #e86e0d;
    color: #fff;
    border: 0;
}
[data-testid='stExpander'] {
    border-radius: 12px;
}

/* Prediction results. */
.result {
    background: linear-gradient(125deg, #eafff1, #f6fffb);
    border: 1px solid #ccebd8;
    border-radius: 18px;
    padding: 25px;
    margin-top: 14px;
}
.result .number {
    color: #197849;
    font-size: 2.7rem;
    font-weight: 800;
    line-height: 1.25;
}
.result .label {
    color: #2b6f48;
    font-size: .94rem;
    font-weight: 700;
}
.empty {
    padding: 24px;
    border: 1px dashed #cbd5e1;
    border-radius: 16px;
    text-align: center;
    color: #64748b;
}

@media (max-width: 900px) {
    [data-testid='stMainBlockContainer'], .block-container {
        padding-top: 4.5rem !important;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
}
@media (max-width: 640px) {
    [data-testid='stMainBlockContainer'], .block-container {
        padding-top: 4rem !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .hero { padding: 24px 20px; border-radius: 16px; }
    .hero h1 { font-size: 1.7rem; }
    .hero p { font-size: .9rem; }
    [data-testid='stForm'] { padding: 16px; }
    .result .number { font-size: 2rem; }
}
</style>
''')

with st.sidebar:
    st.html('''
    <div class='sidebar-brand-card'>
      <div class='sidebar-brand-row'>
        <div class='sidebar-logo'>🛵</div>
        <div class='sidebar-brand-name'>Delivery<span>ETA</span></div>
      </div>
      <p class='sidebar-brand-subtitle'>SMART DELIVERY TIME ESTIMATION<br>Powered by Machine Learning</p>
    </div>
    <div class='sidebar-section-title'>About the project</div>
    <div class='sidebar-info-card'>
      <p>Explore estimated food delivery times using distance, order timing,
      rider details, traffic and weather conditions.</p>
    </div>
    <div class='sidebar-section-title'>Built with</div>
    <div class='sidebar-chip-row'>
      <span class='sidebar-chip'>Python</span>
      <span class='sidebar-chip'>Streamlit</span>
      <span class='sidebar-chip'>FastAPI</span>
      <span class='sidebar-chip'>scikit-learn</span>
      <span class='sidebar-chip'>Docker</span>
    </div>
    <div class='sidebar-note'>
      <strong>ℹ️ Portfolio demonstration</strong><br>
      Uses a historical ML model. Not affiliated with Swiggy or connected to
      live order tracking.
    </div>
    <div class='sidebar-section-title'>Contact</div>
    <div class='sidebar-contact-card'>
      <a class='sidebar-contact-link'
         href='https://www.linkedin.com/in/rajeev-nayan-tripathi-1499581b7/'
         target='_blank' rel='noopener noreferrer'>🔗 LinkedIn Profile</a>
      <a class='sidebar-contact-link'
         href='mailto:rajeevnayantripathi36@gmail.com'>✉️ rajeevnayantripathi36@gmail.com</a>
    </div>
    <div class='sidebar-footer'>DELIVERYETA • ML PORTFOLIO PROJECT</div>
    ''')

st.html('''
<section class='hero'>
  <div class='kicker'>FOOD DELIVERY INTELLIGENCE</div>
  <h1>🛵 DeliveryETA</h1>
  <p>Predict food delivery duration with a trained Gradient Boosting model.<br>
  Enter your delivery scenario and get the result from the FastAPI service.</p>
</section>
''')

st.subheader('📝 Delivery Details')
st.caption('Use a sample historical date, or select another date for a hypothetical scenario.')

with st.form('delivery_form', clear_on_submit = False):
    
    st.markdown('**📍 Order and delivery**')
    col1, col2 = st.columns(2)
    with col1:
        distance_km = st.number_input('Delivery distance (km)', min_value = 0.1, max_value = 21.0, value = 3.0, step = 0.5)
        weather = st.selectbox('Weather', ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Stormy', 'Sandstorms'])
    
    with col2:
        order_type = st.selectbox('Order type', ['Meal', 'Snack', 'Drinks', 'Buffet'])
        traffic = st.selectbox('Traffic', ['Low', 'Medium', 'High', 'Jam'], index = 3)

    st.markdown('**🕒 Order and pickup time**')
    st.caption('Enter both clock times. The app calculates pickup duration, including midnight crossings.')
    
    c1, c2, c3 = st.columns(3)
    with c1:
        order_date = st.date_input('Order date', value = date(2026, 4, 11))
    with c2:
        order_time = st.time_input('Order placed at', value = time(21, 30), step = 60)
    with c3:
        order_picked_time = st.time_input('Order picked at', value = time(21, 45), step = 60)

    with st.expander('⚙️ Additional delivery details', expanded=True):
        
        left, right = st.columns(2)
        
        with left:
            rider_age = st.number_input('Rider age', min_value = 20, max_value = 39, value = 28, step = 1)
            rider_ratings = st.slider('Rider rating', min_value = 2.5, max_value = 5.0, value = 4.5, step = 0.1)
            multiple_deliveries = st.selectbox('Multiple deliveries', [0, 1, 2, 3])
        
        with right:
            vehicle_type = st.selectbox('Vehicle type', ['Motor-Cycle', 'Scooter', 'Electric-Scooter', 'Bicycle'])
            vehicle_condition = st.selectbox('Vehicle condition (dataset code)', [0, 1, 2, 3], index = 2)
            is_festival = st.radio('Festival day?', ['Yes', 'No'], horizontal = True)

    submitted = st.form_submit_button('✨ Predict Delivery Time', use_container_width = True)

if submitted:
    raw_data = {
        'order_date': order_date,
        'order_time': order_time,
        'order_picked_time': order_picked_time,
        'distance_km': float(distance_km),
        'weather': weather,
        'traffic': traffic,
        'order_type': order_type,
        'rider_age': int(rider_age),
        'rider_ratings': float(rider_ratings),
        'multiple_deliveries': int(multiple_deliveries),
        'vehicle_type': vehicle_type,
        'vehicle_condition': int(vehicle_condition),
        'is_festival': int(is_festival == 'Yes')
    }

    st.session_state.pop('prediction_result', None)
    st.session_state.pop('prediction_error', None)
    st.session_state.pop('submitted_payload', None)

    # STEP 1: Validate and engineer the user's original delivery inputs.
    # Validation errors here happen BEFORE the API is called.
    try:
        raw_df = pd.DataFrame([raw_data])
        api_payload = prepare_prediction_json(raw_df)
    except ValueError as exc:
        st.session_state['prediction_error'] = str(exc)
    else:
        # STEP 2: Send the valid 24-feature payload to FastAPI.
        st.session_state['submitted_payload'] = api_payload
        try:
            with st.spinner('Generating prediction...'):
                response = requests.post(
                    API_URL,
                    json=api_payload,
                    timeout=30
                )

            if response.status_code == 422:
                st.session_state['prediction_error'] = (
                    'The API rejected these delivery details (HTTP 422). '
                    'Please check the input values and try again.'
                )
            else:
                response.raise_for_status()
                result = response.json()

                if not isinstance(result, dict):
                    raise TypeError('Unexpected API response format.')

                prediction = float(result['model_prediction'])
                if not math.isfinite(prediction):
                    raise ValueError('Non-finite prediction returned by the API.')

                st.session_state['prediction_result'] = prediction

        except requests.Timeout:
            st.session_state['prediction_error'] = (
                'Prediction timed out. Please try again.'
            )
        except requests.RequestException:
            st.session_state['prediction_error'] = (
                'Prediction service is unavailable. Please try again.'
            )
        except (ValueError, KeyError, TypeError):
            st.session_state['prediction_error'] = (
                'Unexpected response from the prediction service.'
            )

st.subheader('⏱️ Prediction Result')

if st.session_state.get('prediction_error'):
    st.error(st.session_state['prediction_error'])

elif 'prediction_result' in st.session_state:
    
    predicted = st.session_state['prediction_result']
    st.html(f'''
    <section class='result'>
      <div class='label'>✓ Estimated delivery duration</div>
      <div class='number'>{round(predicted)} minutes</div>
      <div style='color:#597463;'>Predicted by the connected ML model</div>
    </section>
    ''')
    payload = st.session_state['submitted_payload']
    m1, m2, m3 = st.columns(3)
    m1.metric('Distance', '{} km'.format(payload['distance_km']))
    m2.metric('Traffic', payload['traffic'])
    m3.metric('Weather', payload['weather'])

elif 'submitted_payload' not in st.session_state:
    st.html('''<div class='empty'>🕒 No prediction yet. Fill the form and click Predict Delivery Time.</div>''')

if 'submitted_payload' in st.session_state:
    with st.expander('🔎 View API payload'):
        st.json(st.session_state['submitted_payload'])
        st.download_button(
            'Download scenario JSON',
            data=json.dumps(st.session_state['submitted_payload'], indent=2),
            file_name='deliveryeta_scenario.json',
            mime='application/json'
        )

st.caption('DeliveryETA • Independent Machine Learning Portfolio Project')
