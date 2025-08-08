import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
from io import BytesIO
import os
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #43a047;
        margin-top: 1rem;
    }
    .card {
        border-radius: 5px;
        background-color: #f5f5f5;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk loading data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Fungsi untuk menampilkan wordcloud
def plot_wordcloud(text, title, width=800, height=400):
    wordcloud = WordCloud(width=width, height=height, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    st.pyplot(fig)

# Fungsi untuk menampilkan metrics dalam bentuk kartu
def display_metric_card(title, value, delta=None, delta_color="normal", help_text=None):
    col1, col2 = st.columns([1, 4])
    with col1:
        if delta is not None:
            st.metric(title, f"{value:.4f}", delta=delta)
        else:
            st.metric(title, f"{value:.4f}")
    with col2:
        if help_text:
            st.info(help_text)

# Fungsi untuk konversi dataframe ke CSV download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Sidebar untuk navigasi
st.sidebar.title("BERT Sentiment Analysis")
page = st.sidebar.selectbox(
    "Navigasi",
    ["Beranda", "Dataset Mentah", "Dataset Berlabel", "Preprocessing", 
     "Model & Training", "Hasil & Evaluasi", "Visualisasi", "Kesimpulan"]
)

# Load data
try:
    df_raw = load_data("data/raw_data.csv")
    df_labeled = load_data("data/labeled_data.csv") 
    
    # Hasil metrics (hardcoded untuk demo, seharusnya dari file hasil)
    results = {
        'accuracy': 0.879,
        'precision': 0.881,
        'recall': 0.879,
        'f1': 0.878,
        'learning_rate': 3e-5,
        'batch_size': 16,
        'epochs': 3,
        'model': 'indonesian-bert-base'
    }
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Demo mode aktif dengan data sampel")
    
    # Generate dummy data jika file tidak tersedia
    df_raw = pd.DataFrame({
        'text': ["Kualitas pelayanannya bagus sekali", 
                "Saya kecewa dengan produk ini", 
                "Biasa saja menurut saya"],
        'timestamp': pd.date_range(start='2023-01-01', periods=3)
    })
    
    df_labeled = pd.DataFrame({
        'text': ["Kualitas pelayanannya bagus sekali", 
                "Saya kecewa dengan produk ini", 
                "Biasa saja menurut saya"],
        'clean_text': ["kualitas pelayanan bagus sekali", 
                      "saya kecewa dengan produk ini", 
                      "biasa saja menurut saya"],
        'sentiment': [2, 0, 1]
    })
    
    results = {
        'accuracy': 0.879,
        'precision': 0.881,
        'recall': 0.879,
        'f1': 0.878,
        'learning_rate': 3e-5,
        'batch_size': 16,
        'epochs': 3,
        'model': 'indonesian-bert-base'
    }

# Konten halaman
if page == "Beranda":
    st.markdown('<p class="main-header">Analisis Sentimen Menggunakan BERT</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Abstrak & Latar Belakang</p>', unsafe_allow_html=True)
        
        abstract = st.text_area(
            "Tuliskan abstrak penelitian Anda di sini",
            height=200,
            value="""Penelitian ini menggunakan model BERT (Bidirectional Encoder Representations from Transformers) untuk analisis sentimen pada data teks bahasa Indonesia. Model ini mampu memahami konteks kata dalam kalimat secara bidireksional, memberikan hasil yang lebih akurat dibandingkan metode tradisional. Studi ini bertujuan untuk mengklasifikasikan sentimen menjadi tiga kategori: negatif, netral, dan positif.""",
            key="abstract"
        )
        
        # Tombol untuk menyimpan abstrak ke file
        if st.button("Simpan Abstrak"):
            with open("abstract.txt", "w") as f:
                f.write(abstract)
            st.success("Abstrak berhasil disimpan!")
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Ringkasan Hasil")
        st.markdown(f"**Akurasi:** {results['accuracy']:.4f}")
        st.markdown(f"**F1-Score:** {results['f1']:.4f}")
        st.markdown(f"**Model:** {results['model']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Info Dataset")
        st.markdown(f"**Jumlah Data:** {len(df_labeled)}")
        if 'sentiment' in df_labeled.columns:
            sentiments = df_labeled['sentiment'].value_counts()
            st.markdown(f"**Negatif:** {sentiments.get(0, 0)}")
            st.markdown(f"**Netral:** {sentiments.get(1, 0)}")
            st.markdown(f"**Positif:** {sentiments.get(2, 0)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<p class="sub-header">Highlight Penelitian</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Preprocessing")
        st.markdown("✅ Normalisasi Teks")
        st.markdown("✅ Pembersihan Karakter Khusus")
        st.markdown("✅ Penghapusan Stopwords")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Model")
        st.markdown("✅ BERT untuk Bahasa Indonesia")
        st.markdown("✅ Fine-tuning dengan Data Spesifik")
        st.markdown("✅ Hyperparameter Optimization")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Evaluasi")
        st.markdown("✅ Accuracy & F1-score")
        st.markdown("✅ Confusion Matrix")
        st.markdown("✅ Cross-validation")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Dataset Mentah":
    st.markdown('<p class="main-header">Dataset Mentah</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Dataset mentah berisi data yang belum diproses dan belum diberi label sentimen.
    Data ini perlu diolah lebih lanjut sebelum dapat digunakan untuk pelatihan model.
    """)
    
    # Upload dataset mentah baru
    st.markdown("### Upload Dataset Baru")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.success("File berhasil diunggah!")
    
    # Eksplorasi dataset
    st.markdown("### Eksplorasi Dataset")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Informasi Dataset")
        st.markdown(f"**Jumlah Data:** {len(df_raw)}")
        st.markdown(f"**Jumlah Kolom:** {len(df_raw.columns)}")
        
        # Check for missing values
        missing_values = df_raw.isnull().sum().sum()
        st.markdown(f"**Missing Values:** {missing_values}")
    
    with col2:
        st.markdown("#### Sample Data")
        st.dataframe(df_raw.head())
    
    # Statistik deskriptif
    st.markdown("### Statistik Deskriptif")
    if 'text' in df_raw.columns:
        text_lengths = df_raw['text'].str.len()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Panjang Teks")
            st.markdown(f"**Min:** {text_lengths.min()}")
            st.markdown(f"**Max:** {text_lengths.max()}")
            st.markdown(f"**Rata-rata:** {text_lengths.mean():.2f}")
            st.markdown(f"**Median:** {text_lengths.median()}")
        
        with col2:
            fig = px.histogram(text_lengths, title='Distribusi Panjang Teks')
            st.plotly_chart(fig, use_container_width=True)
    
    # Download dataset
    st.markdown("### Download Dataset")
    csv = convert_df_to_csv(df_raw)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='raw_dataset.csv',
        mime='text/csv',
    )

elif page == "Dataset Berlabel":
    st.markdown('<p class="main-header">Dataset Berlabel</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Dataset berlabel berisi data yang telah dibersihkan dan diberi label sentimen.
    Data ini siap digunakan untuk pelatihan model.
    """)
    
    # Upload dataset berlabel baru
    st.markdown("### Upload Dataset Baru")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        df_labeled = pd.read_csv(uploaded_file)
        st.success("File berhasil diunggah!")
    
    # Eksplorasi dataset
    st.markdown("### Eksplorasi Dataset")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Informasi Dataset")
        st.markdown(f"**Jumlah Data:** {len(df_labeled)}")
        st.markdown(f"**Jumlah Kolom:** {len(df_labeled.columns)}")
        
        # Distribusi label
        if 'sentiment' in df_labeled.columns:
            sentiments = df_labeled['sentiment'].value_counts()
            st.markdown("#### Distribusi Label")
            st.markdown(f"**Negatif (0):** {sentiments.get(0, 0)}")
            st.markdown(f"**Netral (1):** {sentiments.get(1, 0)}")
            st.markdown(f"**Positif (2):** {sentiments.get(2, 0)}")
    
    with col2:
        st.markdown("#### Sample Data")
        st.dataframe(df_labeled.head())
    
    # Visualisasi distribusi label
    st.markdown("### Distribusi Label Sentimen")
    if 'sentiment' in df_labeled.columns:
        sentiment_counts = df_labeled['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentimen', 'Jumlah']
        sentiment_counts['Sentimen'] = sentiment_counts['Sentimen'].map({0: 'Negatif', 1: 'Netral', 2: 'Positif'})
        
        fig = px.pie(sentiment_counts, values='Jumlah', names='Sentimen', 
                    title='Distribusi Sentimen',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    # Filter data
    st.markdown("### Filter Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'sentiment' in df_labeled.columns:
            sentiment_filter = st.multiselect(
                'Filter berdasarkan sentimen',
                options=[0, 1, 2],
                format_func=lambda x: {0: 'Negatif', 1: 'Netral', 2: 'Positif'}[x],
                default=[0, 1, 2]
            )
        else:
            sentiment_filter = []
    
    with col2:
        search_term = st.text_input('Cari kata dalam teks')
    
    # Apply filters
    filtered_df = df_labeled
    if sentiment_filter:
        filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]
    if search_term:
        filtered_df = filtered_df[filtered_df['text'].str.contains(search_term, case=False)]
    
    st.markdown("### Data Hasil Filter")
    st.dataframe(filtered_df)
    
    # Download dataset
    st.markdown("### Download Dataset")
    csv = convert_df_to_csv(df_labeled)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='labeled_dataset.csv',
        mime='text/csv',
    )

elif page == "Preprocessing":
    st.markdown('<p class="main-header">Preprocessing Data</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Preprocessing data adalah tahapan penting untuk mempersiapkan teks sebelum digunakan dalam model BERT.
    Tahapan ini termasuk pembersihan teks, normalisasi, dan tokenisasi.
    """)
    
    # Preprocessing Steps
    st.markdown('<p class="sub-header">Tahapan Preprocessing</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 1. Pembersihan Teks")
        st.code('''
# Menghapus karakter khusus
def clean_text(text):
    # Hapus URL
    text = re.sub(r'http\S+', '', text)
    # Hapus karakter HTML
    text = re.sub(r'<.*?>', '', text)
    # Hapus karakter non-alfanumerik
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text
        ''')
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 2. Normalisasi")
        st.code('''
# Mengubah teks ke lowercase
def normalize_text(text):
    # Lowercase
    text = text.lower()
    # Mengganti singkatan umum
    text = text.replace('tdk', 'tidak')
    text = text.replace('yg', 'yang')
    text = text.replace('dgn', 'dengan')
    # Dan normalisasi lainnya...
    return text
        ''')
        st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 3. Penghapusan Stopwords")
        st.code('''
# Menghapus stopwords
def remove_stopwords(text):
    # Menggunakan Sastrawi stopwords
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    
    words = text.split()
    filtered_words = [word for word in words 
                     if word not in stopwords]
    return ' '.join(filtered_words)
        ''')
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 4. Tokenisasi BERT")
        st.code('''
# Tokenisasi untuk BERT
def tokenize_text(text, tokenizer):
    tokens = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return tokens
        ''')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Demo preprocessing
    st.markdown('<p class="sub-header">Demo Preprocessing</p>', unsafe_allow_html=True)
    
    # Text input
    demo_text = st.text_area(
        "Masukkan teks untuk preprocessing",
        value="Pelayanannya bagus banget!! Saya sangat puas dengan produk ini... https://example.com #recommended",
        height=100
    )
    
    if st.button("Proses Teks"):
        # Simulated preprocessing steps
        cleaned_text = re.sub(r'http\S+', '', demo_text)
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        normalized_text = cleaned_text.lower()
        
        # Simulate stopword removal (just for demo)
        sample_stopwords = ['dengan', 'ini', 'dan', 'di', 'ke', 'pada', 'untuk']
        words = normalized_text.split()
        filtered_words = [word for word in words if word not in sample_stopwords]
        final_text = ' '.join(filtered_words)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Teks Original")
            st.write(demo_text)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Setelah Pembersihan")
            st.write(cleaned_text)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Hasil Final")
            st.write(final_text)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Sample dari dataset
    if 'clean_text' in df_labeled.columns and 'text' in df_labeled.columns:
        st.markdown('<p class="sub-header">Contoh dari Dataset</p>', unsafe_allow_html=True)
        
        sample_idx = np.random.randint(0, len(df_labeled))
        sample = df_labeled.iloc[sample_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Teks Original")
            st.write(sample['text'])
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Teks Setelah Preprocessing")
            st.write(sample['clean_text'])
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model & Training":
    st.markdown('<p class="main-header">Model & Proses Training</p>', unsafe_allow_html=True)
    
    st.markdown("""
    BERT (Bidirectional Encoder Representations from Transformers) adalah model bahasa yang dilatih 
    untuk memahami makna kata berdasarkan konteksnya dalam kalimat. Model ini dilatih untuk tugas 
    analisis sentimen melalui fine-tuning.
    """)
    
    # Model Architecture
    st.markdown('<p class="sub-header">Arsitektur Model</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Model BERT yang digunakan adalah **indonesian-bert-base** dengan:
        - 12 layer encoder
        - 12 attention heads
        - 768 hidden size
        - 110M parameter
        
        Untuk analisis sentimen, model BERT dimodifikasi dengan menambahkan classification head:
        - Dropout layer (p=0.1)
        - Linear layer untuk klasifikasi (768 → 3)
        """)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Model Parameters")
        st.markdown(f"**Pre-trained Model:** {results['model']}")
        st.markdown(f"**Max Sequence Length:** 128")
        st.markdown(f"**Attention Heads:** 12")
        st.markdown(f"**Hidden Layers:** 12")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Training Configuration
    st.markdown('<p class="sub-header">Konfigurasi Training</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### Hyperparameters")
        st.markdown(f"**Learning Rate:** {results['learning_rate']}")
        st.markdown(f"**Batch Size:** {results['batch_size']}")
        st.markdown(f"**Epochs:** {results['epochs']}")
        st.markdown(f"**Optimizer:** AdamW")
        st.markdown(f"**Weight Decay:** 0.01")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### Teknik Training")
        st.markdown("✅ Early Stopping")
        st.markdown("✅ Learning Rate Scheduler")
        st.markdown("✅ Gradient Accumulation")
        st.markdown("✅ Weight Balancing")
        st.markdown("✅ Mixed Precision Training")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### Data Split")
        st.markdown("✅ 80% Training")
        st.markdown("✅ 10% Validation")
        st.markdown("✅ 10% Testing")
        st.markdown("✅ Stratified Split")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Hyperparameter Tuning
    st.markdown('<p class="sub-header">Hyperparameter Tuning</p>', unsafe_allow_html=True)
    
    # Data untuk plot hyperparameter tuning
    hp_data = {
        'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
        'f1_score': [0.84, 0.86, 0.878, 0.85],
        'accuracy': [0.85, 0.855, 0.879, 0.853]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hp_data['learning_rate'],
        y=hp_data['f1_score'],
        mode='lines+markers',
        name='F1 Score'
    ))
    fig.add_trace(go.Scatter(
        x=hp_data['learning_rate'],
        y=hp_data['accuracy'],
        mode='lines+markers',
        name='Accuracy'
    ))
    fig.update_layout(
        title='Pengaruh Learning Rate terhadap Performa Model',
        xaxis_title='Learning Rate',
        yaxis_title='Score',
        xaxis=dict(type='log')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Training Process
    st.markdown('<p class="sub-header">Proses Training</p>', unsafe_allow_html=True)
    
    # Data untuk loss curves
    epochs = list(range(1, results['epochs'] + 1))
    train_loss = [0.67, 0.42, 0.31]
    val_loss = [0.54, 0.39, 0.32]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines+markers',
        name='Training Loss'
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode='lines+markers',
        name='Validation Loss'
    ))
    fig.update_layout(
        title='Kurva Loss selama Training',
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Hasil & Evaluasi":
    st.markdown('<p class="main-header">Hasil & Evaluasi</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Evaluasi model dilakukan dengan berbagai metrik untuk mengukur performa 
    klasifikasi sentimen pada data test.
    """)
    
    # Metrics
    st.markdown('<p class="sub-header">Metrik Evaluasi</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "Accuracy",
            results['accuracy'],
            help_text="Persentase prediksi yang benar dari total prediksi"
        )
    
    with col2:
        display_metric_card(
            "Precision",
            results['precision'],
            help_text="Rasio true positives terhadap total predicted positives"
        )
    
    with col3:
        display_metric_card(
            "Recall",
            results['recall'],
            help_text="Rasio true positives terhadap total actual positives"
        )
    
    with col4:
        display_metric_card(
            "F1 Score",
            results['f1'],
            help_text="Rata-rata harmonik precision dan recall"
        )
    
    # Confusion Matrix
    st.markdown('<p class="sub-header">Confusion Matrix</p>', unsafe_allow_html=True)
    
    # Data untuk confusion matrix
    conf_matrix = np.array([
        [95, 8, 2],
        [5, 85, 7],
        [3, 10, 95]
    ])
    
    fig = px.imshow(
        conf_matrix,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Negatif', 'Netral', 'Positif'],
        y=['Negatif', 'Netral', 'Positif'],
        text_auto=True,
        color_continuous_scale='Blues'
    )
    fig.update_layout(title='Confusion Matrix')
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown('<p class="sub-header">Classification Report</p>', unsafe_allow_html=True)
    
    # Data untuk classification report
    report_data = {
        'class': ['Negatif', 'Netral', 'Positif', 'Macro Avg', 'Weighted Avg'],
        'precision': [0.92, 0.83, 0.91, 0.887, 0.881],
        'recall': [0.91, 0.87, 0.88, 0.887, 0.879],
        'f1-score': [0.915, 0.85, 0.894, 0.886, 0.878],
        'support': [105, 97, 108, 310, 310]
    }
    report_df = pd.DataFrame(report_data)
    
    st.dataframe(report_df, use_container_width=True)
    
    # Error Analysis
    st.markdown('<p class="sub-header">Analisis Error</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Contoh Kesalahan Prediksi")
        error_examples = pd.DataFrame({
            'Text': [
                'Produknya lumayan, tapi pengirimannya lama.',
                'Saya tidak kecewa dengan pelayanannya.',
                'Barang sampai dengan selamat, tapi packingnya kurang rapi.'
            ],
            'Actual': ['Netral', 'Positif', 'Positif'],
            'Predicted': ['Negatif', 'Negatif', 'Netral']
        })
        st.dataframe(error_examples, use_container_width=True)
    
    with col2:
        st.markdown("#### Kategori Error")
        error_types = pd.DataFrame({
            'Error Type': ['Negasi', 'Sarkasme', 'Sentimen Campuran', 'Konteks Implisit'],
            'Percentage': [35, 25, 30, 10]
        })
        
        fig = px.pie(
            error_types, 
            values='Percentage', 
            names='Error Type',
            title='Distribusi Tipe Error'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Visualisasi":
    st.markdown('<p class="main-header">Visualisasi</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Visualisasi membantu memahami karakteristik dataset dan hasil model 
    secara lebih intuitif.
    """)
    
    # Word Cloud
    st.markdown('<p class="sub-header">Word Cloud</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Semua Data", "Sentimen Negatif", "Sentimen Netral", "Sentimen Positif"])
    
    # Generate some text for word clouds
    all_text = " ".join(df_labeled['clean_text'].tolist())
    
    with tab1:
        plot_wordcloud(all_text, "Word Cloud - Semua Data")
    
    with tab2:
        if 'sentiment' in df_labeled.columns:
            neg_text = " ".join(df_labeled[df_labeled['sentiment'] == 0]['clean_text'].tolist())
            plot_wordcloud(neg_text, "Word Cloud - Sentimen Negatif")
    
    with tab3:
        if 'sentiment' in df_labeled.columns:
            neu_text = " ".join(df_labeled[df_labeled['sentiment'] == 1]['clean_text'].tolist())
            plot_wordcloud(neu_text, "Word Cloud - Sentimen Netral")
    
    with tab4:
        if 'sentiment' in df_labeled.columns:
            pos_text = " ".join(df_labeled[df_labeled['sentiment'] == 2]['clean_text'].tolist())
            plot_wordcloud(pos_text, "Word Cloud - Sentimen Positif")
    
    # Word Frequency
    st.markdown('<p class="sub-header">Frekuensi Kata</p>', unsafe_allow_html=True)
    
    # Generate some word frequency data
    from collections import Counter
    
    words = all_text.split()
    word_counts = Counter(words).most_common(20)
    words, counts = zip(*word_counts)
    
    word_freq_df = pd.DataFrame({
        'Kata': words,
        'Frekuensi': counts
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            word_freq_df,
            x='Kata',
            y='Frekuensi',
            title='Top 20 Kata dalam Dataset',
            color='Frekuensi',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Tabel Frekuensi Kata")
        st.dataframe(word_freq_df, use_container_width=True)
    
    # Sentimen Distribution Over Time (if timestamp available)
    st.markdown('<p class="sub-header">Distribusi Sentimen</p>', unsafe_allow_html=True)
    
    # Generate sample data for sentiment distribution
    sentiment_dist = pd.DataFrame({
        'Kategori': ['Negatif', 'Netral', 'Positif'],
        'Jumlah': [120, 110, 130]
    })
    
    fig = px.bar(
        sentiment_dist,
        x='Kategori',
        y='Jumlah',
        title='Distribusi Sentimen',
        color='Kategori',
        color_discrete_map={
            'Negatif': '#FF6B6B',
            'Netral': '#FFD166',
            'Positif': '#06D6A0'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Attention Visualization
    st.markdown('<p class="sub-header">Visualisasi Attention</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Visualisasi attention menunjukkan kata-kata mana yang diberikan perhatian lebih
    oleh model BERT saat melakukan prediksi sentimen.
    """)
    
    # Sample visualization data
    sample_text = "pelayanan sangat bagus dan harga produk terjangkau"
    attention_weights = [0.02, 0.15, 0.35, 0.05, 0.03, 0.25, 0.15]
    words = sample_text.split()
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create colored text based on attention weights
    colored_text = []
    for word, weight in zip(words, attention_weights):
        colored_text.append(f'<span style="background-color:rgba(255, 165, 0, {weight*2})">{word}</span>')
    
    st.markdown("#### Attention Weights Visualization")
    st.markdown(f'<p style="font-size:20px; line-height:2.5">{"  ".join(colored_text)}</p>', unsafe_allow_html=True)
    
    # Bar chart of attention weights
    attention_df = pd.DataFrame({
        'Kata': words,
        'Attention': attention_weights
    })
    
    fig = px.bar(
        attention_df,
        x='Kata',
        y='Attention',
        title='Attention Weights untuk Setiap Kata',
        color='Attention',
        color_continuous_scale='Oranges'
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Kesimpulan":
    st.markdown('<p class="main-header">Kesimpulan Penelitian</p>', unsafe_allow_html=True)
    
    # Conclusion input
    conclusion = st.text_area(
        "Tuliskan kesimpulan penelitian Anda di sini",
        height=250,
        value="""Penelitian ini menunjukkan bahwa model BERT berhasil diimplementasikan untuk analisis sentimen teks bahasa Indonesia dengan performa yang baik. Berikut kesimpulan utama:

1. Model mencapai akurasi 87.9% dan F1-score 87.8% pada dataset test, menunjukkan kemampuan yang baik dalam mengklasifikasikan sentimen.

2. Hyperparameter terbaik adalah learning rate 3e-5 dengan batch size 16 dan 3 epoch pelatihan.

3. Preprocessing teks seperti normalisasi dan penghapusan stopwords memberikan kontribusi signifikan dalam meningkatkan performa model.

4. Model masih mengalami kesulitan dengan kasus negasi dan sentimen campuran, yang menjadi area potensial untuk perbaikan di penelitian selanjutnya.""",
        key="conclusion"
    )
    
    # Button to save conclusion
    if st.button("Simpan Kesimpulan"):
        with open("conclusion.txt", "w") as f:
            f.write(conclusion)
        st.success("Kesimpulan berhasil disimpan!")
    
    # Limitations and future work
    st.markdown('<p class="sub-header">Batasan & Penelitian Lanjutan</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Batasan Penelitian")
        limitations = st.text_area(
            "Tuliskan batasan penelitian",
            height=150,
            value="""1. Dataset terbatas dalam ukuran dan keberagaman topik.
2. Model kesulitan memahami konteks budaya spesifik dan bahasa informal/slang.
3. Tidak menangani multilingualisme dalam teks.
4. Sentimen campuran (mixed sentiment) masih sulit dideteksi dengan akurat.""",
            key="limitations"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Penelitian Lanjutan")
        future_work = st.text_area(
            "Tuliskan arah penelitian lanjutan",
            height=150,
            value="""1. Menggunakan model multilingual atau model yang lebih besar (BERT-Large).
2. Menambahkan mekanisme khusus untuk mendeteksi negasi dan sentimen campuran.
3. Menerapkan augmentasi data untuk meningkatkan kinerja pada kelas minoritas.
4. Mengembangkan model ensemble yang mengkombinasikan BERT dengan pendekatan lain.""",
            key="future_work"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Final thoughts
    st.markdown('<p class="sub-header">Refleksi Akhir</p>', unsafe_allow_html=True)
    
    final_thoughts = st.text_area(
        "Refleksi tentang penelitian dan pembelajaran",
        height=150,
        value="""Penelitian ini menunjukkan potensi besar model BERT untuk analisis sentimen bahasa Indonesia. Meskipun terdapat beberapa tantangan, model ini menunjukkan performa yang jauh lebih baik dibandingkan pendekatan tradisional.

Dalam pengembangan model NLP untuk bahasa Indonesia ke depan, penting untuk memperhatikan karakteristik khusus bahasa, termasuk variasi dialek dan penggunaan bahasa informal yang umum di media sosial. Kolaborasi antara ahli bahasa dan praktisi machine learning akan sangat berharga untuk memajukan bidang ini.""",
        key="final_thoughts"
    )
    
    # References
    st.markdown('<p class="sub-header">Referensi</p>', unsafe_allow_html=True)
    
    references = st.text_area(
        "Tuliskan referensi penelitian",
        height=150,
        value="""1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

2. Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., ... & Purwarianti, A. (2020). IndoNLU: Benchmark and resources for evaluating Indonesian natural language understanding. arXiv preprint arXiv:2009.05387.

3. Koto, F., Rahimi, A., Lau, J. H., & Baldwin, T. (2020). IndoBERT: A pre-trained language model for Indonesian. arXiv preprint arXiv:2009.05387.""",
        key="references"
    )