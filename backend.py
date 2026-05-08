from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re
import emoji
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sklearn.svm import LinearSVC

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

app = Flask(__name__)
CORS(app)

# ── Chargement des modèles ──────────────────────────────────
print("Chargement des modèles...")
tfidf  = joblib.load('data/checkpoints/tfidf.joblib')
splits = joblib.load('data/checkpoints/splits.joblib')

svm = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42, C=1.0)
svm.fit(splits['X_train_tfidf'], splits['y_train'])
print("✅ Modèles chargés")

# ── Preprocessing ──────────────────────────────────────────
def preprocess(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_explain(text, top_n=8):
    clean = preprocess(text)
    vec   = tfidf.transform([clean])
    pred  = int(svm.predict(vec)[0])

    feature_names = np.array(tfidf.get_feature_names_out())
    coefs         = svm.coef_[0]
    vec_arr       = vec.toarray()[0]
    active_idx    = np.where(vec_arr > 0)[0]

    scores = [(feature_names[i], float(coefs[i] * vec_arr[i])) for i in active_idx]
    scores.sort(key=lambda x: abs(x[1]), reverse=True)

    toxic_words  = [[w, s] for w, s in scores if s > 0][:top_n]
    normal_words = [[w, s] for w, s in scores if s < 0][:top_n]

    return pred, toxic_words, normal_words, clean

# ── YouTube helpers ────────────────────────────────────────
def extract_video_id(url):
    for p in [r'(?:v=|\/)([0-9A-Za-z_-]{11})', r'youtu\.be\/([0-9A-Za-z_-]{11})']:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def fetch_comments(video_id, max_results=100):
    yt   = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    resp = yt.commentThreads().list(
        part='snippet', videoId=video_id,
        maxResults=max_results, textFormat='plainText', order='relevance'
    ).execute()
    return [{'author': i['snippet']['topLevelComment']['snippet']['authorDisplayName'],
             'text'  : i['snippet']['topLevelComment']['snippet']['textDisplay'],
             'likes' : i['snippet']['topLevelComment']['snippet']['likeCount']}
            for i in resp.get('items', [])]

def get_video_info(video_id):
    try:
        yt   = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        resp = yt.videos().list(part='snippet,statistics', id=video_id).execute()
        if resp['items']:
            it = resp['items'][0]
            return {'title'    : it['snippet']['title'],
                    'channel'  : it['snippet']['channelTitle'],
                    'views'    : int(it['statistics'].get('viewCount', 0)),
                    'thumbnail': it['snippet']['thumbnails']['medium']['url']}
    except: pass
    return None

# ── ROUTES ────────────────────────────────────────────────
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Texte vide'}), 400

    pred, toxic_words, normal_words, preprocessed = predict_explain(text)
    return jsonify({
        'pred'        : pred,
        'toxic_words' : toxic_words,
        'normal_words': normal_words,
        'preprocessed': preprocessed
    })

@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube():
    data     = request.json
    url      = data.get('url', '').strip()
    video_id = extract_video_id(url)

    if not video_id:
        return jsonify({'error': 'URL YouTube invalide'}), 400

    try:
        video_info = get_video_info(video_id)
        comments   = fetch_comments(video_id, max_results=100)
    except Exception as e:
        return jsonify({'error': f'Erreur API YouTube : {str(e)}'}), 500

    if not comments:
        return jsonify({'error': 'Aucun commentaire trouvé ou commentaires désactivés'}), 404

    results = []
    for c in comments:
        pred, tw, nw, _ = predict_explain(c['text'])
        results.append({
            'author'      : c['author'],
            'text'        : c['text'],
            'likes'       : c['likes'],
            'pred'        : pred,
            'toxic_words' : [w for w, _ in tw],
            'normal_words': [w for w, _ in nw],
        })

    return jsonify({'video_info': video_info, 'comments': results})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'LinearSVC + TF-IDF'})

if __name__ == '__main__':
    app.run(debug=True, port=8000)