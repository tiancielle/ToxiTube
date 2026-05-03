import streamlit as st
import joblib
import numpy as np
import re
import emoji
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
import pandas as pd

# ─── Configuration ───────────────────────────────────────────
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

st.set_page_config(
    page_title="ToxiTube",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;600;700&family=Cabinet+Grotesk:wght@300;400;500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --cream: #FAF7F2;
    --warm-white: #FFFDF9;
    --coral: #FF6B6B;
    --coral-light: #FFE8E8;
    --coral-dark: #E54444;
    --teal: #0FA3B1;
    --teal-light: #E0F5F7;
    --amber: #F7B731;
    --amber-light: #FFF8E1;
    --green: #26C485;
    --green-light: #E6FBF3;
    --slate: #2D3748;
    --slate-light: #718096;
    --border: #E8E0D5;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--slate);
}

/* Header */
.hero {
    background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #F7B731 100%);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.1);
    border-radius: 50%;
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    color: white;
    letter-spacing: 0.05em;
    line-height: 1;
    margin: 0;
    text-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.hero-sub {
    color: rgba(255,255,255,0.85);
    font-size: 1.1rem;
    font-weight: 300;
    margin-top: 0.5rem;
    letter-spacing: 0.02em;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Cards */
.card {
    background: var(--warm-white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}

.metric-card {
    background: var(--warm-white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    line-height: 1;
    letter-spacing: 0.02em;
}

.metric-label {
    color: var(--slate-light);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* Comment cards */
.comment-card {
    background: var(--warm-white);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    border: 1px solid var(--border);
    border-left: 5px solid;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}

.toxic-card   { border-left-color: var(--coral); background: linear-gradient(to right, #FFF5F5, var(--warm-white)); }
.normal-card  { border-left-color: var(--green); background: linear-gradient(to right, #F0FDF8, var(--warm-white)); }

.badge-toxic {
    background: var(--coral-light);
    color: var(--coral-dark);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid #FFCDD2;
}

.badge-normal {
    background: var(--green-light);
    color: #0D9B68;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid #B2DFDB;
}

/* Word chips */
.chip-toxic {
    background: var(--coral-light);
    color: var(--coral-dark);
    border: 1px solid #FFCDD2;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 2px;
    display: inline-block;
    font-weight: 500;
}

.chip-normal {
    background: var(--green-light);
    color: #0D9B68;
    border: 1px solid #B2DFDB;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 2px;
    display: inline-block;
    font-weight: 500;
}

/* Result box */
.result-toxic {
    background: linear-gradient(135deg, #FF6B6B, #FF8E53);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px rgba(255,107,107,0.3);
}

.result-normal {
    background: linear-gradient(135deg, #26C485, #0FA3B1);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px rgba(38,196,133,0.3);
}

.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }

.result-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 0.05em;
}

/* Video info */
.video-info {
    background: var(--warm-white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    gap: 1rem;
    align-items: center;
}

/* Gauge bar */
.gauge-bar-bg {
    background: #F0EBE3;
    border-radius: 20px;
    height: 12px;
    margin: 0.5rem 0;
    overflow: hidden;
}

/* Streamlit overrides */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: var(--warm-white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--slate) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--coral) !important;
    box-shadow: 0 0 0 3px rgba(255,107,107,0.12) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #FF6B6B, #FF8E53) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 15px rgba(255,107,107,0.3) !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,107,107,0.4) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--warm-white);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border);
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #FF6B6B, #FF8E53) !important;
    color: white !important;
}

.stRadio > div { flex-direction: row; gap: 1rem; }

div[data-testid="stDecoration"] { display: none; }

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 0.05em;
    color: var(--slate);
    margin-bottom: 1rem;
}

.author-line {
    color: var(--slate-light);
    font-size: 0.82rem;
    margin-bottom: 0.4rem;
}

.comment-text {
    color: var(--slate);
    font-size: 0.92rem;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧪 NLP Project — Text Mining</div>
    <div class="hero-title">ToxiTube</div>
    <div class="hero-sub">Détecteur de toxicité pour commentaires YouTube · LinearSVC + TF-IDF · DistilBERT</div>
</div>
""", unsafe_allow_html=True)

# ─── Chargement modèles ──────────────────────────────────────
@st.cache_resource
def load_models():
    tfidf = joblib.load('data/checkpoints/tfidf.joblib')
    splits = joblib.load('data/checkpoints/splits.joblib')
    from sklearn.svm import LinearSVC
    svm = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42, C=1.0)
    svm.fit(splits['X_train_tfidf'], splits['y_train'])
    return tfidf, svm

with st.spinner('Chargement des modèles...'):
    try:
        tfidf, svm = load_models()
        st.success('✅ Modèles chargés — LinearSVC + TF-IDF')
    except Exception as e:
        st.error(f'❌ Erreur : {e}')
        st.stop()

# ─── Preprocessing ───────────────────────────────────────────
def preprocess(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_and_explain(text, top_n=8):
    clean = preprocess(text)
    vec = tfidf.transform([clean])
    pred = svm.predict(vec)[0]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = svm.coef_[0]
    vec_arr = vec.toarray()[0]
    active_idx = np.where(vec_arr > 0)[0]
    scores = [(feature_names[i], coefs[i] * vec_arr[i]) for i in active_idx]
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return pred, [(w,s) for w,s in scores if s > 0][:top_n], [(w,s) for w,s in scores if s < 0][:top_n]

# ─── YouTube ─────────────────────────────────────────────────
def extract_video_id(url):
    for p in [r'(?:v=|\/)([0-9A-Za-z_-]{11})', r'youtu\.be\/([0-9A-Za-z_-]{11})']:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def fetch_comments(video_id, max_results=50):
    try:
        yt = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        resp = yt.commentThreads().list(
            part='snippet', videoId=video_id,
            maxResults=max_results, textFormat='plainText', order='relevance'
        ).execute()
        return [{'author': i['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                 'text':   i['snippet']['topLevelComment']['snippet']['textDisplay'],
                 'likes':  i['snippet']['topLevelComment']['snippet']['likeCount']}
                for i in resp.get('items', [])]
    except Exception as e:
        st.error(f"Erreur API YouTube : {e}")
        return []

def get_video_info(video_id):
    try:
        yt = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        resp = yt.videos().list(part='snippet,statistics', id=video_id).execute()
        if resp['items']:
            it = resp['items'][0]
            return {'title': it['snippet']['title'],
                    'channel': it['snippet']['channelTitle'],
                    'views': int(it['statistics'].get('viewCount', 0)),
                    'thumbnail': it['snippet']['thumbnails']['medium']['url']}
    except: pass
    return None

# ─── TABS ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎬  Analyser une vidéo YouTube", "✍️  Tester un commentaire"])

# ════════════════════════════════════════
# TAB 1 — YouTube
# ════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Collez une URL YouTube</div>', unsafe_allow_html=True)

    url = st.text_input("", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")
    analyse_btn = st.button("🔍 Analyser la vidéo", key="yt_btn")

    if analyse_btn and url:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ URL invalide — vérifiez le lien YouTube.")
        else:
            with st.spinner("Récupération des commentaires via l'API YouTube..."):
                info = get_video_info(video_id)
                comments = fetch_comments(video_id, max_results=50)

            if not comments:
                st.warning("⚠️ Aucun commentaire trouvé ou commentaires désactivés.")
            else:
                # Infos vidéo
                if info:
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.image(info['thumbnail'], use_container_width=True)
                    with c2:
                        st.markdown(f"### {info['title']}")
                        st.markdown(f"📺 **{info['channel']}** &nbsp;·&nbsp; 👁️ {info['views']:,} vues")

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # Analyse des commentaires
                with st.spinner(f"Analyse de {len(comments)} commentaires..."):
                    results = []
                    for c in comments:
                        pred, tw, nw = predict_and_explain(c['text'])
                        results.append({**c, 'pred': pred, 'toxic_words': tw, 'normal_words': nw})

                df_res = pd.DataFrame(results)
                n_toxic  = int((df_res['pred'] == 1).sum())
                n_normal = int((df_res['pred'] == 0).sum())
                pct = n_toxic / len(df_res) * 100

                # Métriques
                m1, m2, m3 = st.columns(3)
                color = "#FF6B6B" if pct > 30 else "#F7B731" if pct > 10 else "#26C485"
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:{color}">{pct:.0f}%</div>
                        <div class="gauge-bar-bg">
                            <div style="height:12px; width:{pct}%; background:{color}; border-radius:20px;
                                        transition:width 1s ease;"></div>
                        </div>
                        <div class="metric-label">Taux de toxicité</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#FF6B6B">{n_toxic}</div>
                        <div class="metric-label">☠️ Commentaires toxiques</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#26C485">{n_normal}</div>
                        <div class="metric-label">✅ Commentaires normaux</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # Filtre
                col_f, _ = st.columns([2, 3])
                with col_f:
                    filtre = st.radio("Afficher :", ["Tous", "Toxiques", "Normaux"], horizontal=True)

                filtered = df_res
                if filtre == "Toxiques":
                    filtered = df_res[df_res['pred'] == 1]
                elif filtre == "Normaux":
                    filtered = df_res[df_res['pred'] == 0]

                st.markdown(f"**{len(filtered)} commentaires affichés**")
                st.markdown("")

                for _, row in filtered.iterrows():
                    is_toxic = row['pred'] == 1
                    card_cls = "toxic-card" if is_toxic else "normal-card"
                    badge    = '<span class="badge-toxic">☠️ Toxique</span>' if is_toxic else '<span class="badge-normal">✅ Normal</span>'

                    words_html = ""
                    if is_toxic and row['toxic_words']:
                        for w, _ in row['toxic_words'][:5]:
                            words_html += f'<span class="chip-toxic">⚠ {w}</span>'
                    elif not is_toxic and row['normal_words']:
                        for w, _ in row['normal_words'][:4]:
                            words_html += f'<span class="chip-normal">✓ {w}</span>'

                    text_preview = str(row['text'])[:280] + ('...' if len(str(row['text'])) > 280 else '')

                    st.markdown(f"""
                    <div class="comment-card {card_cls}">
                        <div class="author-line">
                            <span>👤 <b>{row['author']}</b></span>
                            &nbsp;·&nbsp;
                            <span>👍 {row['likes']}</span>
                            <span style="float:right">{badge}</span>
                        </div>
                        <div class="comment-text">{text_preview}</div>
                        <div style="margin-top:0.5rem">{words_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 2 — Texte libre
# ════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Testez un commentaire</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "",
        placeholder="Collez ou tapez un commentaire ici...\nEx: You are such an idiot, nobody watches your videos!",
        height=130,
        label_visibility="collapsed"
    )

    predict_btn = st.button("🔍 Analyser ce commentaire", key="txt_btn")

    if predict_btn and user_text.strip():
        pred, toxic_words, normal_words = predict_and_explain(user_text)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if pred == 1:
            st.markdown("""
            <div class="result-toxic">
                <div class="result-emoji">☠️</div>
                <div class="result-label">Commentaire Toxique</div>
                <div style="opacity:0.85; font-size:0.9rem; margin-top:0.3rem;">
                    Ce commentaire a été classifié comme toxique par notre modèle
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-normal">
                <div class="result-emoji">✅</div>
                <div class="result-label">Commentaire Normal</div>
                <div style="opacity:0.85; font-size:0.9rem; margin-top:0.3rem;">
                    Ce commentaire a été classifié comme non-toxique par notre modèle
                </div>
            </div>""", unsafe_allow_html=True)

        # Explication
        st.markdown('<div class="section-title">Explication du modèle</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="card">
                <div style="font-weight:600; color:#FF6B6B; margin-bottom:0.8rem;">
                    ⚠️ Mots qui poussent vers TOXIQUE
                </div>
            """, unsafe_allow_html=True)
            if toxic_words:
                for w, s in toxic_words:
                    st.markdown(f'<span class="chip-toxic">⚠ {w} &nbsp;<b>{s:+.3f}</b></span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#aaa; font-size:0.9rem">Aucun détecté</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="card">
                <div style="font-weight:600; color:#26C485; margin-bottom:0.8rem;">
                    ✓ Mots qui poussent vers NORMAL
                </div>
            """, unsafe_allow_html=True)
            if normal_words:
                for w, s in normal_words:
                    st.markdown(f'<span class="chip-normal">✓ {w} &nbsp;<b>{s:+.3f}</b></span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#aaa; font-size:0.9rem">Aucun détecté</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Texte original**")
            st.markdown(f'<div class="card"><div class="comment-text">{user_text}</div></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown("**Texte preprocessé**")
            st.markdown(f'<div class="card"><div class="comment-text" style="color:#718096; font-family:monospace; font-size:0.85rem">{preprocess(user_text)}</div></div>', unsafe_allow_html=True)
