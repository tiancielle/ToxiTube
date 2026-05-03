import streamlit as st
import joblib
import numpy as np
import re
import emoji
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ff4444, #ff8800);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    color: #888;
    font-size: 1.1rem;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
}

.metric-label {
    color: #888;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.comment-card {
    background: #1a1a1a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    border-left: 4px solid;
    font-size: 0.92rem;
    line-height: 1.5;
}

.toxic-card   { border-left-color: #ff4444; }
.normal-card  { border-left-color: #22c55e; }

.badge-toxic  { background:#ff4444; color:white; padding:2px 8px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-normal { background:#22c55e; color:white; padding:2px 8px; border-radius:20px; font-size:0.75rem; font-weight:600; }

.word-chip-pos { background:#ff444422; color:#ff8888; border:1px solid #ff444444;
                 padding:3px 10px; border-radius:20px; font-size:0.8rem; margin:3px; display:inline-block; }
.word-chip-neg { background:#22c55e22; color:#4ade80; border:1px solid #22c55e44;
                 padding:3px 10px; border-radius:20px; font-size:0.8rem; margin:3px; display:inline-block; }

.stTextInput > div > div > input, .stTextArea textarea {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #f0f0f0 !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #ff4444, #ff8800) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
}

.divider { border-top: 1px solid #222; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement des modèles ───────────────────────────────────
@st.cache_resource
def load_models():
    tfidf = joblib.load('data/checkpoints/tfidf.joblib')
    splits = joblib.load('data/checkpoints/splits.joblib')

    from sklearn.svm import LinearSVC
    X_train_tfidf = splits['X_train_tfidf']
    y_train = splits['y_train']

    svm = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42, C=1.0)
    svm.fit(X_train_tfidf, y_train)

    return tfidf, svm

# ─── Preprocessing ───────────────────────────────────────────
def preprocess(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── Prédiction + explication ────────────────────────────────
def predict_and_explain(text, tfidf, svm, top_n=10):
    clean = preprocess(text)
    vec = tfidf.transform([clean])
    pred = svm.predict(vec)[0]

    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = svm.coef_[0]
    vec_arr = vec.toarray()[0]
    active_idx = np.where(vec_arr > 0)[0]

    scores = [(feature_names[i], coefs[i] * vec_arr[i]) for i in active_idx]
    scores.sort(key=lambda x: abs(x[1]), reverse=True)

    toxic_words  = [(w, s) for w, s in scores if s > 0][:top_n]
    normal_words = [(w, s) for w, s in scores if s < 0][:top_n]

    return pred, toxic_words, normal_words

# ─── YouTube API ─────────────────────────────────────────────
def extract_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})'
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def fetch_comments(video_id, max_results=100):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results,
            textFormat='plainText',
            order='relevance'
        ).execute()
        comments = []
        for item in response.get('items', []):
            text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            likes = item['snippet']['topLevelComment']['snippet']['likeCount']
            comments.append({'author': author, 'text': text, 'likes': likes})
        return comments
    except Exception as e:
        st.error(f"Erreur API YouTube : {e}")
        return []

def get_video_info(video_id):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        response = youtube.videos().list(part='snippet,statistics', id=video_id).execute()
        if response['items']:
            item = response['items'][0]
            return {
                'title'   : item['snippet']['title'],
                'channel' : item['snippet']['channelTitle'],
                'views'   : int(item['statistics'].get('viewCount', 0)),
                'thumbnail': item['snippet']['thumbnails']['medium']['url']
            }
    except:
        pass
    return None

# ─── UI ──────────────────────────────────────────────────────
st.markdown('<p class="main-title">🧪 ToxiTube</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Détecteur de toxicité pour commentaires YouTube — NLP Project</p>', unsafe_allow_html=True)

# Chargement modèles
with st.spinner('Chargement des modèles...'):
    try:
        tfidf, svm = load_models()
        st.success(' Modèles chargés (LinearSVC + TF-IDF)')
    except Exception as e:
        st.error(f' Erreur chargement modèles : {e}')
        st.stop()

# ─── Tabs ────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎬 Analyser une vidéo YouTube", "✍️ Tester un commentaire"])

# ════════════════════════════════════════════════════════════
# TAB 1 — YouTube
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Collez l'URL d'une vidéo YouTube")
    url = st.text_input("URL YouTube", placeholder="https://www.youtube.com/watch?v=...")

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyse_btn = st.button("🔍 Analyser", key="yt_btn")

    if analyse_btn and url:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("URL invalide — vérifiez le lien.")
        else:
            with st.spinner("Récupération des commentaires..."):
                info = get_video_info(video_id)
                comments = fetch_comments(video_id, max_results=50)

            if not comments:
                st.warning("Aucun commentaire trouvé ou commentaires désactivés.")
            else:
                # Infos vidéo
                if info:
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.image(info['thumbnail'])
                    with c2:
                        st.markdown(f"**{info['title']}**")
                        st.markdown(f"📺 {info['channel']} &nbsp;|&nbsp; 👁️ {info['views']:,} vues")

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # Analyse
                with st.spinner(f"Analyse de {len(comments)} commentaires..."):
                    results = []
                    for c in comments:
                        pred, tw, nw = predict_and_explain(c['text'], tfidf, svm)
                        results.append({**c, 'pred': pred, 'toxic_words': tw, 'normal_words': nw})

                df_res = pd.DataFrame(results)
                n_toxic  = (df_res['pred'] == 1).sum()
                n_normal = (df_res['pred'] == 0).sum()
                pct = n_toxic / len(df_res) * 100

                # Métriques
                m1, m2, m3 = st.columns(3)
                with m1:
                    color = "#ff4444" if pct > 30 else "#ff8800" if pct > 10 else "#22c55e"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:{color}">{pct:.0f}%</div>
                        <div class="metric-label">Taux de toxicité</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#ff4444">{n_toxic}</div>
                        <div class="metric-label">Commentaires toxiques</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#22c55e">{n_normal}</div>
                        <div class="metric-label">Commentaires normaux</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # Filtre
                filtre = st.radio("Afficher :", ["Tous", "Toxiques seulement", "Normaux seulement"], horizontal=True)

                filtered = df_res
                if filtre == "Toxiques seulement":
                    filtered = df_res[df_res['pred'] == 1]
                elif filtre == "Normaux seulement":
                    filtered = df_res[df_res['pred'] == 0]

                st.markdown(f"**{len(filtered)} commentaires affichés**")

                for _, row in filtered.iterrows():
                    is_toxic = row['pred'] == 1
                    card_cls = "toxic-card" if is_toxic else "normal-card"
                    badge    = '<span class="badge-toxic">☠️ Toxique</span>' if is_toxic else '<span class="badge-normal">✅ Normal</span>'

                    # Mots influents
                    words_html = ""
                    if is_toxic and row['toxic_words']:
                        for w, _ in row['toxic_words'][:5]:
                            words_html += f'<span class="word-chip-pos">⚠️ {w}</span>'
                    elif not is_toxic and row['normal_words']:
                        for w, _ in row['normal_words'][:5]:
                            words_html += f'<span class="word-chip-neg">✓ {w}</span>'

                    st.markdown(f"""
                    <div class="comment-card {card_cls}">
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                            <span style="color:#888; font-size:0.85rem;">👤 {row['author']} &nbsp;·&nbsp; 👍 {row['likes']}</span>
                            {badge}
                        </div>
                        <div>{row['text'][:300]}{'...' if len(row['text']) > 300 else ''}</div>
                        <div style="margin-top:0.5rem;">{words_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — Texte libre
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Testez un commentaire")
    user_text = st.text_area(
        "Entrez un commentaire",
        placeholder="Ex: You are such an idiot, nobody likes your videos!",
        height=120
    )

    col_btn2, _ = st.columns([1, 4])
    with col_btn2:
        predict_btn = st.button("🔍 Analyser", key="txt_btn")

    if predict_btn and user_text:
        pred, toxic_words, normal_words = predict_and_explain(user_text, tfidf, svm)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Résultat principal
        if pred == 1:
            st.markdown("""
            <div style="background:#ff444422; border:1px solid #ff444466; border-radius:12px;
                        padding:1.5rem; text-align:center; margin-bottom:1.5rem;">
                <div style="font-size:3rem;">☠️</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#ff4444;">
                    COMMENTAIRE TOXIQUE
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#22c55e22; border:1px solid #22c55e66; border-radius:12px;
                        padding:1.5rem; text-align:center; margin-bottom:1.5rem;">
                <div style="font-size:3rem;">✅</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#22c55e;">
                    COMMENTAIRE NORMAL
                </div>
            </div>""", unsafe_allow_html=True)

        # Explication
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**⚠️ Mots qui poussent vers TOXIQUE**")
            if toxic_words:
                for w, s in toxic_words[:8]:
                    st.markdown(f'<span class="word-chip-pos">⚠️ {w} ({s:+.3f})</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#888">Aucun mot toxique détecté</span>', unsafe_allow_html=True)

        with c2:
            st.markdown("**✓ Mots qui poussent vers NORMAL**")
            if normal_words:
                for w, s in normal_words[:8]:
                    st.markdown(f'<span class="word-chip-neg">✓ {w} ({s:+.3f})</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#888">Aucun mot neutre détecté</span>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**Texte preprocessé :**")
        st.code(preprocess(user_text), language=None)
