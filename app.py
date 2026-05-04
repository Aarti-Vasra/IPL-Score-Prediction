import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark cricket-pitch background */
  .stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1f14 50%, #0d1117 100%);
    color: #e6edf3;
  }

  /* Header */
  .hero {
    text-align: center;
    padding: 2rem 0 1rem 0;
  }
  .hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 4px;
    background: linear-gradient(90deg, #f0c040, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .hero p {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 0.4rem;
  }

  /* Cards */
  .card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
  }
  .card-title {
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #f0c040;
    margin-bottom: 1rem;
    font-weight: 600;
  }

  /* Result box */
  .result-box {
    background: linear-gradient(135deg, #1a2f1a, #162416);
    border: 2px solid #2ea043;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
  }
  .result-label {
    font-size: 0.8rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #56d364;
    margin-bottom: 0.5rem;
  }
  .result-score {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    color: #ffffff;
    line-height: 1;
    letter-spacing: 4px;
  }
  .result-range {
    color: #8b949e;
    font-size: 0.9rem;
    margin-top: 0.4rem;
  }
  .result-range span {
    color: #f0c040;
    font-weight: 600;
  }

  /* Metric cards */
  .metric-row {
    display: flex;
    gap: 12px;
    margin-top: 1rem;
  }
  .metric-box {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
  }
  .metric-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #f0c040;
  }
  .metric-lbl {
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Override streamlit defaults */
  div[data-testid="stSelectbox"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stNumberInput"] label {
    color: #c9d1d9 !important;
    font-weight: 500;
  }
  .stSlider > div > div > div > div {
    background-color: #f0c040 !important;
  }
  div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #f0c040, #ff6b35) !important;
    color: #0d1117 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.4rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
  }
  div[data-testid="stButton"] button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
  }

  /* Hide streamlit branding */
  #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load & Train Model (cached so it runs once) ───────────────────────────────
@st.cache_resource
def load_model():
    # Try to find the CSV in common locations
    possible_paths = [
        'ipl.csv',
        os.path.join(os.path.dirname(__file__), 'ipl.csv'),
        '/mnt/user-data/uploads/ipl.csv',
    ]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break

    if df is None:
        return None, None

    # ----- Same preprocessing as notebook -----
    columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
    df.drop(labels=columns_to_remove, axis=1, inplace=True)

    consistent_teams = [
        'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
        'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
        'Delhi Daredevils', 'Sunrisers Hyderabad'
    ]
    df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
    df = df[df['overs'] >= 5.0]
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

    encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

    cols = ['date',
            'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
            'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
            'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
            'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
            'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
            'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
            'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
            'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
            'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']
    encoded_df = encoded_df[cols]

    X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
    y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
    X_train.drop(labels='date', axis=1, inplace=True)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, consistent_teams


# ── Prediction function ───────────────────────────────────────────────────────
def predict_score(model, batting_team, bowling_team, overs, runs, wickets, runs_last_5, wickets_last_5):
    teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
             'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
             'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

    bat_enc = [1 if t == batting_team else 0 for t in teams]
    bowl_enc = [1 if t == bowling_team else 0 for t in teams]
    features = bat_enc + bowl_enc + [overs, runs, wickets, runs_last_5, wickets_last_5]
    features = np.array([features])

    prediction = int(model.predict(features)[0])
    return prediction


# ── App Layout ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏏 IPL Score Predictor</h1>
  <p>First Innings Score Prediction · Linear Regression Model · Trained on IPL 2008–2016</p>
</div>
""", unsafe_allow_html=True)

model, teams = load_model()

if model is None:
    st.error("❌ Could not find `ipl.csv`. Please place it in the same folder as `app.py` and restart.")
    st.stop()

teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
         'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
         'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

team_short = {
    'Chennai Super Kings': 'CSK 🦁',
    'Delhi Daredevils': 'DD 🔵',
    'Kings XI Punjab': 'KXIP 🦁',
    'Kolkata Knight Riders': 'KKR 💜',
    'Mumbai Indians': 'MI 💙',
    'Rajasthan Royals': 'RR 🩷',
    'Royal Challengers Bangalore': 'RCB 🔴',
    'Sunrisers Hyderabad': 'SRH 🧡',
}

# ── Team Selection ────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">⚔️ Team Selection</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams, index=4)
with col2:
    bowling_options = [t for t in teams if t != batting_team]
    bowling_team = st.selectbox("🎳 Bowling Team", bowling_options, index=0)
st.markdown('</div>', unsafe_allow_html=True)

# ── Match Situation ───────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📊 Current Match Situation</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    overs = st.slider("Overs Played", min_value=5.0, max_value=19.5, value=10.0, step=0.5,
                      help="Minimum 5 overs required for prediction")
with col2:
    runs = st.number_input("Runs Scored", min_value=0, max_value=300, value=80,
                           help="Total runs scored so far")
with col3:
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=2,
                              help="Total wickets lost so far")

col4, col5 = st.columns(2)
with col4:
    runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=150, value=40,
                                   help="Runs scored in the last 5 overs")
with col5:
    wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=5, value=1,
                                      help="Wickets fallen in the last 5 overs")
st.markdown('</div>', unsafe_allow_html=True)

# ── Current Run Rate ──────────────────────────────────────────────────────────
crr = runs / overs if overs > 0 else 0
overs_left = 20 - overs

st.markdown(f"""
<div class="metric-row">
  <div class="metric-box">
    <div class="metric-val">{crr:.2f}</div>
    <div class="metric-lbl">Current Run Rate</div>
  </div>
  <div class="metric-box">
    <div class="metric-val">{overs_left:.1f}</div>
    <div class="metric-lbl">Overs Remaining</div>
  </div>
  <div class="metric-box">
    <div class="metric-val">{10 - wickets}</div>
    <div class="metric-lbl">Wickets in Hand</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("PREDICT FINAL SCORE"):
    if batting_team == bowling_team:
        st.error("Batting and bowling teams cannot be the same!")
    else:
        predicted = predict_score(model, batting_team, bowling_team,
                                  overs, runs, wickets, runs_last_5, wickets_last_5)
        low = predicted - 10
        high = predicted + 5

        bat_s = team_short.get(batting_team, batting_team)
        bowl_s = team_short.get(bowling_team, bowling_team)

        st.markdown(f"""
        <div class="result-box">
          <div class="result-label">✅ Predicted Final Score</div>
          <div class="result-score">{predicted}</div>
          <div class="result-range">
            Expected Range: <span>{low}</span> – <span>{high}</span> runs
          </div>
          <br>
          <div style="color:#8b949e; font-size:0.85rem;">
            {bat_s} vs {bowl_s} &nbsp;·&nbsp; After {overs} overs &nbsp;·&nbsp; {runs}/{wickets}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence message
        if wickets >= 7:
            st.warning("⚠️ Many wickets fallen — score may be lower than predicted.")
        elif crr > 10:
            st.info("🔥 High run rate! This could be a big score.")
        elif overs > 15:
            st.info("📌 Death overs in play — big hits expected!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<div style='text-align:center; color:#30363d; font-size:0.75rem; padding: 1rem 0;'>
  Model trained on IPL Seasons 1–9 (2008–2016) · Tested on Season 10 (2017)
  <br>Linear Regression · sklearn · Streamlit
</div>
""", unsafe_allow_html=True)
