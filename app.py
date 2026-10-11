import os
import re
import nltk
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from urllib.parse import urlparse
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('stopwords', quiet=True)

# Initialize NLP processing tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Suspicious keywords commonly found in spam
SPAM_KEYWORDS = {
    'congratulations': '🎉 Prize/Congratulations',
    'winner': '🏆 Winner',
    'claim': '✋ Claim',
    'prize': '🎁 Prize',
    'free': '💰 Free',
    'urgent': '⚠️ Urgent',
    'act now': '⏰ Act Now',
    'limited time': '⏱️ Limited Time',
    'click here': '🔗 Click Here',
    'verify': '🔐 Verify Account',
    'confirm': '✔️ Confirm',
    'update': '🔄 Update',
    'password': '🔑 Password',
    'credit card': '💳 Credit Card',
    'bank': '🏦 Bank',
    'win': '🎯 Win',
    'million': '💸 Million',
    'inheritance': '💰 Inheritance',
    'nigerian prince': '👑 Nigerian Prince',
    'unsubscribe': '❌ Unsubscribe',
}

def preprocess_text(text: str) -> str:
    """Cleans raw input text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def detect_urls(text: str) -> list:
    """Extract URLs from text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls

def detect_suspicious_keywords(text: str) -> dict:
    """Detect suspicious keywords in text."""
    text_lower = text.lower()
    detected = {}
    for keyword, label in SPAM_KEYWORDS.items():
        if keyword in text_lower:
            detected[keyword] = label
    return detected

def analyze_urls(urls: list) -> list:
    """Identify URL characteristics that deserve extra caution."""
    findings = []
    shorteners = {'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'rb.gy'}
    for url in urls:
        parsed = urlparse(url)
        hostname = (parsed.hostname or '').lower()
        reasons = []
        if parsed.scheme != 'https':
            reasons.append('not HTTPS')
        if hostname in shorteners:
            reasons.append('shortened link')
        if re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}', hostname):
            reasons.append('IP address URL')
        if '@' in parsed.netloc:
            reasons.append('hidden destination')
        findings.append({'url': url, 'host': hostname, 'risk': reasons or ['no obvious URL risk']})
    return findings

def calculate_risk_score(text: str, result: dict, metadata: dict) -> dict:
    """Create a transparent, explainable risk score from detected signals."""
    signals = []
    keyword_points = min(len(result['keywords']) * 7, 35)
    if keyword_points:
        signals.append(('Suspicious language', keyword_points))
    url_findings = analyze_urls(result['urls'])
    risky_urls = sum('no obvious URL risk' not in item['risk'] for item in url_findings)
    url_points = min(len(result['urls']) * 8 + risky_urls * 10, 30)
    if url_points:
        signals.append(('Links and URL safety', url_points))
    lower_text = text.lower()
    urgency_points = 15 if any(word in lower_text for word in ('urgent', 'immediately', 'act now', 'limited time')) else 0
    if urgency_points:
        signals.append(('Urgency pressure', urgency_points))
    money_points = 15 if any(word in lower_text for word in ('money', 'cash', 'credit card', 'bank', 'payment', '$')) else 0
    if money_points:
        signals.append(('Financial request', money_points))
    metadata_points = 5 if metadata.get('sender') and '@' not in metadata['sender'] else 0
    if metadata_points:
        signals.append(('Unusual sender format', metadata_points))
    model_points = int(result['confidence'] * 20) if result['is_spam'] else 0
    if model_points:
        signals.append(('Machine-learning signal', model_points))
    score = min(sum(points for _, points in signals), 100)
    level = 'High' if score >= 60 else 'Medium' if score >= 30 else 'Low'
    return {'score': score, 'level': level, 'signals': signals, 'url_findings': url_findings}

def train_model_from_dataframe(dataframe: pd.DataFrame) -> dict:
    """Train and persist a model from a labeled dataframe."""
    columns = {column.lower(): column for column in dataframe.columns}
    text_column = next((columns[name] for name in ('text', 'email', 'message', 'v2') if name in columns), None)
    label_column = next((columns[name] for name in ('label', 'class', 'category', 'v1') if name in columns), None)
    if not text_column or not label_column:
        raise ValueError("CSV needs text/email/message and label/class columns.")
    training_df = dataframe[[text_column, label_column]].dropna().copy()
    training_df.columns = ['text', 'label']
    training_df['label'] = training_df['label'].astype(str).str.lower().map({'ham': 0, 'legitimate': 0, '0': 0, 'spam': 1, '1': 1})
    training_df = training_df.dropna()
    if len(training_df) < 6 or training_df['label'].nunique() < 2:
        raise ValueError("Add at least 6 rows containing both spam and ham labels.")
    training_df['clean_text'] = training_df['text'].astype(str).apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(
        training_df['clean_text'], training_df['label'], test_size=0.2, random_state=42,
        stratify=training_df['label'] if training_df['label'].value_counts().min() >= 2 else None,
    )
    pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=3000)), ('nb', MultinomialNB())])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, predictions, labels=[0, 1]),
        'test_data': (X_test, y_test),
        'training_rows': len(training_df),
    }
    with open('model_pipeline.pkl', 'wb') as file:
        pickle.dump(pipeline, file)
    with open('model_metrics.pkl', 'wb') as file:
        pickle.dump(metrics, file)
    load_model.clear()
    load_metrics.clear()
    return metrics

def compare_models(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Evaluate three classifiers using the same uploaded labeled dataset."""
    columns = {column.lower(): column for column in dataframe.columns}
    text_column = next((columns[name] for name in ('text', 'email', 'message', 'v2') if name in columns), None)
    label_column = next((columns[name] for name in ('label', 'class', 'category', 'v1') if name in columns), None)
    if not text_column or not label_column:
        raise ValueError("CSV needs text/email/message and label/class columns.")
    data = dataframe[[text_column, label_column]].dropna()
    labels = data[label_column].astype(str).str.lower().map({'ham': 0, 'legitimate': 0, '0': 0, 'spam': 1, '1': 1})
    if labels.isna().any() or labels.nunique() < 2:
        raise ValueError("Labels must contain both spam and ham values.")
    texts = data[text_column].astype(str).apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    classifiers = {'Naive Bayes': MultinomialNB(), 'Logistic Regression': LogisticRegression(max_iter=1000), 'Linear SVM': LinearSVC()}
    rows = []
    for name, classifier in classifiers.items():
        pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=3000)), ('classifier', classifier)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rows.append({'Model': name, 'Accuracy': accuracy_score(y_test, predictions), 'Precision': precision_score(y_test, predictions, zero_division=0), 'Recall': recall_score(y_test, predictions, zero_division=0), 'F1 Score': f1_score(y_test, predictions, zero_division=0)})
    return pd.DataFrame(rows).set_index('Model')

def train_and_save_model(data_path: str = "spam.csv"):
    """Trains a Spam Classifier and serializes the artifacts."""
    if not os.path.exists(data_path):
        mock_data = {
            'text': [
                "WINNER!! You have won a $1000 gift card! Claim now!",
                "Hey, are we still meeting for lunch today?",
                "Free urgent prize entry! Text WIN to 80085",
                "Can you please send me the project reports?",
                "URGENT! Call this number to claim your prize.",
                "Hi, just checking if the meeting is still at 3pm",
                "Congratulations! You are a lucky winner of $50,000",
                "The project deadline has been extended to next Friday",
                "Click here now to verify your account immediately!",
                "Let me know if you need any additional information"
            ],
            'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
        }
        df = pd.DataFrame(mock_data)
    else:
        df = pd.read_csv(data_path, encoding='latin-1')
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

    df['clean_text'] = df['text'].apply(preprocess_text)
    X = df['clean_text']
    y = df['label'].map({'ham': 0, 'spam': 1}) if set(df['label'].unique()).issubset({'ham', 'spam'}) else df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000)),
        ('nb', MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'test_data': (X_test, y_test)
    }

    with open("model_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open("model_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)


@st.cache_resource
def load_model():
    """Load the trained classifier."""
    with open("model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_metrics():
    """Load model metrics."""
    try:
        with open("model_metrics.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

def classify_single_email(text: str, model):
    """Classify a single email and return detailed analysis."""
    processed_text = preprocess_text(text)
    prediction = model.predict([processed_text])[0]
    probabilities = model.predict_proba([processed_text])[0]
    is_spam = prediction == 1 or str(prediction).lower() == 'spam'
    confidence = probabilities[1] if is_spam else probabilities[0]
    
    urls = detect_urls(text)
    keywords = detect_suspicious_keywords(text)
    
    return {
        'is_spam': is_spam,
        'confidence': confidence,
        'urls': urls,
        'keywords': keywords,
        'processed_text': processed_text
    }

def run_streamlit_app():
    """Main Streamlit dashboard."""
    st.set_page_config(
        page_title="Inbox Shield Pro | Advanced Spam Classifier",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        :root { --ink: #102a43; --muted: #52606d; --teal: #0f766e;
                --coral: #e76f51; --line: #d9e2ec; --paper: #ffffff; }
        .stApp { background: linear-gradient(135deg, #f4f8fb 0%, #eef8f5 52%, #fff8f2 100%); }
        .block-container { max-width: 1280px; padding: 2.5rem 2rem 4rem; }
        .hero { position: relative; overflow: hidden; padding: 2rem 2.2rem 2.2rem;
                border-radius: 18px; background: linear-gradient(115deg, #102a43, #0f766e);
                box-shadow: 0 16px 38px rgba(16, 42, 67, .18); margin-bottom: 2.5rem; }
        .hero:after { content: ''; position: absolute; width: 280px; height: 280px;
                      right: -80px; top: -120px; border-radius: 50%;
                      background: rgba(231, 111, 81, .72); }
        .eyebrow { position: relative; z-index: 1; color: #9ff3db; font-size: .82rem;
                   font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
        .hero h1 { position: relative; z-index: 1; color: #ffffff; max-width: 800px;
                   font-size: clamp(2.2rem, 5vw, 3.4rem); line-height: 1.08;
                   margin: .5rem 0 .7rem; }
        .hero p { position: relative; z-index: 1; color: #d8f3ee; max-width: 750px;
                  font-size: 1.08rem; line-height: 1.6; margin: 0; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #102a43, #0d1f2d); }
        [data-testid="stSidebar"] * { color: #edf7f6; }
        [data-testid="stSidebar"] .stAlert { background: rgba(159, 243, 219, .14); border: 1px solid #4fb3a3; }
        .metric-card { background: #ffffff; border: 1.5px solid #d9e2ec; border-radius: 14px; 
                      padding: 1.2rem; box-shadow: 0 3px 12px rgba(0,0,0,.08); }
        .stTextArea textarea { border: 2px solid #c9d6e2; border-radius: 12px; background: #ffffff;
                              color: var(--ink); font-size: 1rem; line-height: 1.55; }
        .stTextArea textarea:focus { border-color: var(--teal); box-shadow: 0 0 0 .2rem rgba(15, 118, 110, .14); }
        .stButton button, .stFormSubmitButton button { border-radius: 10px; border: 0;
            background: var(--coral); color: #ffffff; font-weight: 800; min-height: 2.8rem;
            box-shadow: 0 7px 16px rgba(231, 111, 81, .25); transition: all .2s ease; }
        .stButton button:hover, .stFormSubmitButton button:hover { background: #c95137; transform: translateY(-2px); }
        h2, h3 { color: var(--ink); margin-top: 1.5rem; }
        [data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--line);
            border-radius: 12px; padding: .8rem 1rem; }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, var(--teal), var(--coral)); }
        .keyword-badge { display: inline-block; background: #ffe6cc; color: #c95137; 
                        padding: .35rem .8rem; border-radius: 6px; margin: .3rem .3rem .3rem 0;
                        font-size: .85rem; font-weight: 600; }
        .url-badge { display: inline-block; background: #cce5ff; color: #0051ba; 
                    padding: .35rem .8rem; border-radius: 6px; margin: .3rem .3rem .3rem 0;
                    font-size: .85rem; font-weight: 600; word-break: break-all; }
        [data-testid="stDataFrame"] { overflow-x: auto; }
        @media (max-width: 700px) {
            .block-container { padding: 1rem .75rem 2rem; max-width: 100%; }
            .hero { padding: 1.15rem 1rem 1.25rem; margin-bottom: 1.25rem; border-radius: 14px; }
            .hero:after { width: 150px; height: 150px; right: -65px; top: -55px; }
            .hero h1 { font-size: 2rem; line-height: 1.12; margin: .45rem 0 .6rem; }
            .hero p { font-size: .95rem; line-height: 1.45; }
            .eyebrow { font-size: .7rem; }
            h2 { font-size: 1.35rem; }
            h3 { font-size: 1.1rem; }
            [data-testid="stHorizontalBlock"] { flex-direction: column; gap: .65rem; }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
            [data-testid="stForm"] { padding: .25rem 0; }
            .stTextArea textarea { min-height: 170px; font-size: 16px; }
            .stTextInput input, .stSelectbox select, .stFileUploader section { min-height: 46px; font-size: 16px; }
            .stButton button, .stFormSubmitButton button, .stDownloadButton button { min-height: 48px; width: 100%; font-size: 1rem; }
            [data-testid="stTabs"] [role="tablist"] { overflow-x: auto; scrollbar-width: thin; }
            [data-testid="stTabs"] button { min-height: 46px; white-space: nowrap; padding: 0 .75rem; }
            [data-testid="stMetric"] { padding: .7rem .8rem; }
            [data-testid="stMetricValue"] { font-size: 1.45rem; }
            .metric-card { padding: .9rem; }
            .keyword-badge, .url-badge { display: block; margin: .35rem 0; word-break: break-word; }
            [data-testid="stDataFrame"] div { font-size: .82rem; }
            [data-testid="stSidebar"] { min-width: 280px; }
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize model
    if not os.path.exists("model_pipeline.pkl"):
        with st.spinner("🚀 Preparing the classifier for first run..."):
            train_and_save_model()

    model = load_model()
    metrics = load_metrics()
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    # Hero Section
    st.markdown("""
        <div class="hero">
            <div class="eyebrow">🛡️ Inbox Shield Pro</div>
            <h1>Enterprise-Grade Spam Detection</h1>
            <p>Advanced AI-powered email classification with keyword analysis, URL detection, and batch processing.</p>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛡️ Inbox Shield Pro")
        st.caption("Machine Learning Email Classifier")
        
        if metrics:
            st.success("✅ Model Ready")
            st.divider()
            st.markdown("**📊 Model Performance**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                st.metric("Precision", f"{metrics['precision']:.1%}")
            with col2:
                st.metric("Recall", f"{metrics['recall']:.1%}")
                st.metric("F1 Score", f"{metrics['f1']:.1%}")
        else:
            st.info("Model training in progress...")
        
        st.divider()
        st.markdown("**📖 How It Works**")
        st.markdown("""
        1. **Input**: Paste an email or upload CSV
        2. **Analysis**: Keyword & URL detection
        3. **Prediction**: Spam or Legitimate
        4. **Indicators**: See why it was flagged
        """)
        st.info("⚠️ Demo model. Don't share sensitive data.")

    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Email", "📊 Metrics & Analytics", "📁 Batch Prediction", "🧪 Model Lab"])

    # TAB 1: Single Email Classification
    with tab1:
        st.subheader("Analyze a Single Email")
        with st.form("classifier_form"):
            metadata_col1, metadata_col2 = st.columns(2)
            with metadata_col1:
                sender = st.text_input("Sender address (optional)", placeholder="sender@example.com")
            with metadata_col2:
                subject = st.text_input("Subject line (optional)", placeholder="Your account needs attention")
            user_input = st.text_area(
                "Email Content",
                height=220,
                placeholder="Paste the complete email or SMS here...",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns([3, 1])
            with col1:
                submitted = st.form_submit_button("🔎 Analyze Email", type="primary", use_container_width=True)
            with col2:
                clear_btn = st.form_submit_button("Clear", use_container_width=True)

        if submitted and user_input.strip():
            full_text = f"Subject: {subject}\n{user_input}" if subject.strip() else user_input
            metadata = {'sender': sender.strip(), 'subject': subject.strip()}
            result = classify_single_email(full_text, model)
            risk = calculate_risk_score(full_text, result, metadata)
            st.session_state.prediction_history.insert(0, {
                'Time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Sender': sender or 'Not provided',
                'Subject': subject or 'Not provided',
                'Result': 'SPAM' if result['is_spam'] else 'LEGITIMATE',
                'Confidence': f"{result['confidence'] * 100:.1f}%",
                'Risk': f"{risk['level']} ({risk['score']}/100)",
            })
            st.session_state.prediction_history = st.session_state.prediction_history[:50]
            
            st.divider()
            
            # Main Result
            result_col, confidence_col = st.columns([2, 1.2])
            with result_col:
                if result['is_spam']:
                    st.error("🚫 SPAM DETECTED", icon="🚫")
                    st.write("⚠️ This message shows characteristics of spam. Proceed with caution.")
                else:
                    st.success("✅ LEGITIMATE EMAIL", icon="✅")
                    st.write("✔️ This message appears to be genuine correspondence.")
            
            with confidence_col:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                st.progress(float(result['confidence']))

            risk_col, risk_detail_col = st.columns([1, 2])
            with risk_col:
                st.metric("Explainable Risk", f"{risk['score']}/100")
                st.caption(f"{risk['level']} risk based on message signals")
            with risk_detail_col:
                st.markdown("### Why this score?")
                if risk['signals']:
                    st.dataframe(pd.DataFrame(risk['signals'], columns=['Signal', 'Points']), hide_index=True, use_container_width=True)
                else:
                    st.info("No additional risk signals detected.")
            
            # Analysis Sections
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔍 Suspicious Keywords Detected")
                if result['keywords']:
                    for keyword, label in result['keywords'].items():
                        st.markdown(f"<span class='keyword-badge'>{label}</span>", unsafe_allow_html=True)
                else:
                    st.info("No suspicious keywords detected.")
            
            with col2:
                st.markdown("### 🔗 URLs Found")
                if result['urls']:
                    for url in result['urls']:
                        st.markdown(f"<span class='url-badge'>{url}</span>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame([{'URL': item['url'], 'Host': item['host'], 'Safety findings': ', '.join(item['risk'])} for item in risk['url_findings']]), hide_index=True, use_container_width=True)
                else:
                    st.info("No URLs detected in this email.")
            
            st.divider()
            with st.expander("📝 View Processed Text (for ML model)"):
                st.caption("This is the normalized text sent to the classifier:")
                st.code(result['processed_text'] or "No meaningful words after cleaning.", language="text")
            report = pd.DataFrame([{
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'sender': sender, 'subject': subject, 'prediction': 'spam' if result['is_spam'] else 'legitimate',
                'model_confidence': round(result['confidence'], 4), 'risk_score': risk['score'],
                'risk_level': risk['level'], 'keywords': ', '.join(result['keywords']),
                'urls': ', '.join(result['urls']), 'risk_signals': '; '.join(f'{name}: {points}' for name, points in risk['signals']),
            }])
            st.download_button("📥 Download Analysis Report", report.to_csv(index=False), "email_analysis_report.csv", "text/csv")

        elif submitted:
            st.warning("Please enter a message to analyze.")

    # TAB 2: Model Performance Dashboard
    with tab2:
        st.subheader("📊 Model Performance Metrics")
        
        if metrics:
            # Key Metrics
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            with met_col1:
                st.markdown('<div class="metric-card"><div style="color: #0f766e; font-size: 2rem; font-weight: 800;">⭐</div><div style="color: #52606d; font-size: .8rem; font-weight: 700; text-transform: uppercase;">Accuracy</div><div style="color: #102a43; font-size: 1.8rem; font-weight: 800;">{:.1%}</div></div>'.format(metrics['accuracy']), unsafe_allow_html=True)
            with met_col2:
                st.markdown('<div class="metric-card"><div style="color: #0f766e; font-size: 2rem; font-weight: 800;">📈</div><div style="color: #52606d; font-size: .8rem; font-weight: 700; text-transform: uppercase;">Precision</div><div style="color: #102a43; font-size: 1.8rem; font-weight: 800;">{:.1%}</div></div>'.format(metrics['precision']), unsafe_allow_html=True)
            with met_col3:
                st.markdown('<div class="metric-card"><div style="color: #0f766e; font-size: 2rem; font-weight: 800;">🎯</div><div style="color: #52606d; font-size: .8rem; font-weight: 700; text-transform: uppercase;">Recall</div><div style="color: #102a43; font-size: 1.8rem; font-weight: 800;">{:.1%}</div></div>'.format(metrics['recall']), unsafe_allow_html=True)
            with met_col4:
                st.markdown('<div class="metric-card"><div style="color: #0f766e; font-size: 2rem; font-weight: 800;">✔️</div><div style="color: #52606d; font-size: .8rem; font-weight: 700; text-transform: uppercase;">F1 Score</div><div style="color: #102a43; font-size: 1.8rem; font-weight: 800;">{:.1%}</div></div>'.format(metrics['f1']), unsafe_allow_html=True)
            
            st.divider()
            
            # Confusion Matrix
            st.markdown("### 📊 Confusion Matrix")
            cm = metrics['confusion_matrix']
            cm_text = f"""
            | | Predicted Ham | Predicted Spam |
            |---|---|---|
            | **Actual Ham** | {cm[0,0]} | {cm[0,1]} |
            | **Actual Spam** | {cm[1,0]} | {cm[1,1]} |
            """
            st.markdown(cm_text)
            
            # Interpretation
            st.info(f"""
            ✅ **True Negatives (TN)**: {cm[0,0]} — Correctly identified as legitimate  
            ❌ **False Positives (FP)**: {cm[0,1]} — Incorrectly flagged as spam  
            ❌ **False Negatives (FN)**: {cm[1,0]} — Missed spam emails  
            ✅ **True Positives (TP)**: {cm[1,1]} — Correctly identified as spam
            """)
            if st.session_state.prediction_history:
                st.markdown("### Recent Prediction History")
                history_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(history_df, hide_index=True, use_container_width=True)
                st.download_button("📥 Export History", history_df.to_csv(index=False), "prediction_history.csv", "text/csv")
                if st.button("Clear History"):
                    st.session_state.prediction_history = []
                    st.rerun()
            st.markdown("### Signal Overview")
            st.bar_chart(pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'], 'Score': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]}).set_index('Metric'))
        else:
            st.warning("Metrics not available. Train the model first.")

    # TAB 3: Batch CSV Prediction
    with tab3:
        st.subheader("📁 Batch Email Prediction")
        st.write("Upload a CSV file with emails to classify them in bulk.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"📋 Loaded {len(df)} rows")
                
                # Find the email column
                email_cols = [col for col in df.columns if 'email' in col.lower() or 'text' in col.lower() or 'message' in col.lower()]
                
                if email_cols:
                    email_column = st.selectbox("Select the email/text column", email_cols)
                    
                    if st.button("🚀 Run Batch Prediction", type="primary"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for idx, row in df.iterrows():
                            email_text = str(row[email_column])
                            result = classify_single_email(email_text, model)
                            results.append({
                                'Email': email_text[:50] + '...' if len(email_text) > 50 else email_text,
                                'Prediction': '🚫 SPAM' if result['is_spam'] else '✅ HAM',
                                'Confidence': f"{result['confidence']*100:.1f}%",
                                'Keywords': ', '.join(result['keywords'].keys())[:40] if result['keywords'] else 'None',
                                'URLs': len(result['urls'])
                            })
                            progress_bar.progress((idx + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        summary = results_df['Prediction'].value_counts().rename_axis('Prediction').to_frame('Count')
                        st.markdown("### Batch Summary")
                        st.bar_chart(summary)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="spam_predictions.csv",
                            mime="text/csv",
                            type="primary"
                        )
                else:
                    st.error("Could not find an email/text column. Please ensure your CSV has a column named 'email', 'text', or 'message'.")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # TAB 4: Custom training and model comparison
    with tab4:
        st.subheader("🧪 Model Lab")
        st.write("Upload labeled data to retrain the classifier or compare multiple algorithms.")
        st.caption("Accepted columns: text/email/message plus label/class. Labels should be spam and ham.")
        training_file = st.file_uploader("Upload labeled training CSV", type=['csv'], key='training_csv')
        if training_file:
            try:
                training_df = pd.read_csv(training_file)
                st.write(f"📋 Training dataset: {len(training_df)} rows")
                st.dataframe(training_df.head(8), hide_index=True, use_container_width=True)
                lab_col1, lab_col2 = st.columns(2)
                with lab_col1:
                    if st.button("🧠 Retrain Active Model", type="primary", use_container_width=True):
                        with st.spinner("Training and evaluating the new model..."):
                            new_metrics = train_model_from_dataframe(training_df)
                        st.success(f"Model updated using {new_metrics['training_rows']} labeled messages.")
                        st.rerun()
                with lab_col2:
                    if st.button("⚖️ Compare Algorithms", use_container_width=True):
                        with st.spinner("Evaluating Naive Bayes, Logistic Regression, and Linear SVM..."):
                            comparison = compare_models(training_df)
                        st.session_state.model_comparison = comparison
                if 'model_comparison' in st.session_state:
                    st.markdown("### Model Comparison")
                    comparison = st.session_state.model_comparison
                    st.dataframe(comparison.style.format('{:.1%}'), use_container_width=True)
                    st.bar_chart(comparison[['Accuracy', 'F1 Score']])
                    st.download_button("📥 Download Comparison", comparison.to_csv(), "model_comparison.csv", "text/csv")
            except Exception as error:
                st.error(f"Could not process the training file: {error}")
        else:
            st.info("Upload a labeled CSV to activate custom training and model comparison.")

if __name__ == "__main__":
    run_streamlit_app()