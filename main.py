# Add import for hashing filter state
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# ---------- Load and Prepare Dataset ----------
df = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\movie\imdb_2024_entire_cleaned_data.csv')

# Clean missing values
for col in ['rating', 'duration', 'genre', 'voting', 'storyline']:
    df[col] = df[col].fillna('N/A')

# Convert rating and duration to numeric for filtering
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['duration_minutes'] = df['duration'].str.extract(r'(\d+)').astype(float).fillna(0)

# ---------- Vectorize Storylines ----------
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['storyline'])

# ---------- Streamlit App ----------
st.set_page_config(page_title="IMDB Movie Search & Recommender", layout="wide")
st.title("\U0001F3A5 IMDB Movie Search & Recommender")

# Inject custom CSS styling
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        .info-plate-container {
            width: fit-content;
            max-width: 100%;
            height: 50vh;
            overflow-y: auto;
            overflow-x: hidden;
            border: 1px solid #ccc;
            border-radius: 12px;
            padding: 10px;
            background-color: #f9f9f9;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: center;
        }
        .info-plate {
            width: 600px;
            border: 1px solid #ccc;
            border-radius: 12px;
            padding: 20px;
            font-family: Arial, sans-serif;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            background: linear-gradient(135deg, #f9f9f9, #ffffff);
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 15px;
        }
        .info-plate:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        .movie-name {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            position: relative;
        }
        .movie-name::after {
            content: '';
            position: absolute;
            left: 0;
            bottom: -5px;
            width: 50px;
            height: 3px;
            background-color: #f5b301;
            transition: width 0.3s ease;
        }
        .info-plate:hover .movie-name::after {
            width: 100%;
        }
        .details-row {
            display: flex;
            gap: 15px;
            margin-bottom: 12px;
        }
        .detail-item {
            font-size: 14px;
            color: #666;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .score-icon {
            color: #f5b301;
            font-size: 14px;
        }
        .genre-voting-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .genre {
            font-size: 14px;
            color: #444;
            font-style: italic;
        }
        .votes-count {
            background-color: #eee;
            color: #000;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .votes-count i {
            color: #666;
        }
        .storyline {
            font-size: 14px;
            margin-top: 10px;
            background: #ffffff;
            color: #000;
            padding: 10px;
            border-left: 4px solid #ccc;
            border-radius: 6px;
            font-family: 'Georgia', serif;
            font-style: normal;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar Filters ----------
st.sidebar.header("\U0001F50D Filter Movies")

min_rating, max_rating = st.sidebar.slider("Rating Range", 0.0, 10.0, (0.0, 10.0), 0.1)
min_duration, max_duration = st.sidebar.slider("Duration (minutes)", 0, 300, (0, 300), 10)
selected_genre = st.sidebar.multiselect("Select Genre", options=sorted(set(df['genre'].dropna())), default=[])
search_title = st.sidebar.text_input("Search by Movie Title")

# ---------- Main Search Area ----------
st.header("\U0001F9E0 Advanced Search by Storyline")
user_input = st.text_area("Enter a story or keywords")

# ---------- Filtering Function ----------
def filter_dataframe(df):
    filtered = df[
        (df['rating'] >= min_rating) &
        (df['rating'] <= max_rating) &
        (df['duration_minutes'] >= min_duration) &
        (df['duration_minutes'] <= max_duration)
    ]
    if selected_genre:
        filtered = filtered[filtered['genre'].isin(selected_genre)]
    if search_title:
        filtered = filtered[filtered['title'].str.contains(search_title, case=False, na=False)]
    return filtered

# Apply filters
filtered_df = filter_dataframe(df)

# ---------- Session State for Lazy Loading ----------
current_filter_hash = hashlib.md5(f"{min_rating}{max_rating}{min_duration}{max_duration}{selected_genre}{search_title}".encode()).hexdigest()

if 'filter_hash' not in st.session_state or st.session_state.filter_hash != current_filter_hash:
    st.session_state.filter_hash = current_filter_hash
    st.session_state.num_shown = 10

# ---------- Recommendation System ----------
if user_input:
    user_vector = vectorizer.transform([user_input])
    cos_sim = cosine_similarity(user_vector, tfidf_matrix)
    similarity_scores = cos_sim[0]

    filtered_indices = filtered_df.index.tolist()
    filtered_scores = similarity_scores[filtered_indices]

    if filtered_scores.max() < 0.05:
        st.warning("No related movie found with your filters. Try adjusting them or use different keywords.")

    sorted_indices = [filtered_indices[i] for i in filtered_scores.argsort()[::-1][:5]]

    st.subheader("\U0001F50E Top 5 Similar Movies")
    for idx in sorted_indices:
        result = df.iloc[idx]
        score = similarity_scores[idx]

        html_code = f"""
        <div class=\"info-plate-container\">
            <div class=\"info-plate\">
                <div class=\"movie-name\">{result['title']}</div>
                <div class=\"details-row\">
                    <span class=\"detail-item\">
                        <i class=\"fas fa-film\"></i>
                        Rating: {result['rating']}
                    </span>
                    <span class=\"detail-item\">
                        <i class=\"fas fa-clock\"></i>
                        {result['duration']}
                    </span>
                    <span class=\"detail-item\">
                        <i class=\"fas fa-star score-icon\"></i>
                        Score: {score:.2f}/10
                    </span>
                </div>
                <div class=\"genre-voting-row\">
                    <div class=\"genre\">
                        <i class=\"fas fa-tags\"></i>
                        {result['genre']}
                    </div>
                    <div class=\"votes-count\">
                        <i class=\"fas fa-thumbs-up\"></i>
                        Votes: {result['voting']}
                    </div>
                </div>
                <div class=\"storyline\">
                    <i class=\"fas fa-quote-left\"></i>
                    {result['storyline']}
                </div>
            </div>
        </div>
        <br>
        """
        st.markdown(html_code, unsafe_allow_html=True)
        st.markdown("---")

elif not filtered_df.empty:
    st.subheader("\U0001F39E Movies Based on Filters")

    for _, result in filtered_df.head(st.session_state.num_shown).iterrows():
        html_code = f"""
        <div class=\"info-plate-container\">
            <div class=\"info-plate\">
                <div class=\"movie-name\">{result['title']}</div>
                <div class=\"details-row\">
                    <span class=\"detail-item\">
                        <i class=\"fas fa-film\"></i>
                        Rating: {result['rating']}
                    </span>
                    <span class=\"detail-item\">
                        <i class=\"fas fa-clock\"></i>
                        {result['duration']}
                    </span>
                </div>
                <div class=\"genre-voting-row\">
                    <div class=\"genre\">
                        <i class=\"fas fa-tags\"></i>
                        {result['genre']}
                    </div>
                    <div class=\"votes-count\">
                        <i class=\"fas fa-thumbs-up\"></i>
                        Votes: {result['voting']}
                    </div>
                </div>
                <div class=\"storyline\">
                    <i class=\"fas fa-quote-left\"></i>
                    {result['storyline']}
                </div>
            </div>
        </div>
        <br>
        """
        st.markdown(html_code, unsafe_allow_html=True)
        st.markdown("---")

    remaining = len(filtered_df) - st.session_state.num_shown
    if remaining > 0:
        if st.button(f"Show More"):
            st.session_state.num_shown += 10
else:
    st.info("Use the sidebar to filter movies or enter a storyline to search.")
