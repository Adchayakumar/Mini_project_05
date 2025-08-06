# 🎬 IMDB Movie Recommendation System

This project is a **content-based movie recommendation system** that suggests movies to users based on the **storyline they enter**. It uses **Natural Language Processing (NLP)** techniques like **TF-IDF vectorization** and **Cosine Similarity** to find and display the most relevant movie matches from the IMDB database.

> ✅ This project was developed as part of a course to demonstrate practical skills in data collection, processing, and similarity-based recommendation systems.

---

## 🚀 Features

- 🧠 Recommend movies based on user-inputted storyline
- 🎯 Filter by Rating, Duration, Genre, and Movie Title
- 🪄 Lazy Loading for Filtered Results (Show More Button)
- 🎨 Custom UI built with **Streamlit** and enhanced with **HTML/CSS**
- 🌐 Real-world data scraped from **IMDB** using **Selenium**
- 🎭 Cleaned and processed movie data from **23 genres**
- ⚡ Efficient similarity matching using **TF-IDF** + **Cosine Similarity**


---

## 📁 Dataset Description

The dataset was **extracted from the IMDB website using Selenium** and contains the following columns:

- `title`: Movie name  
- `storyline`: Short description or plot  
- `genre`: Movie genre(s)  
- `rating`: IMDB rating (float)  
- `voting`: Number of votes  
- `duration`: Movie duration

> The cleaned dataset is saved as: `imdb_2024_entire_cleaned_data.csv` (included in this repository)

---

## 🧼 Preprocessing Steps

1. Combined multiple CSVs (one per genre) into one dataset.
2. Cleaned:
   - Movie titles (removed symbols, numbering)
   - Converted `rating` to float
   - Cleaned `voting` column (removed brackets/extra symbols)
   - Validated `duration` format
3. Merged genres for duplicate titles.
4. Dropped invalid or missing rows for accurate results.

---

## 🧠 Recommendation Logic

1. User enters a custom storyline.
2. Input is vectorized using **TF-IDF** (Term Frequency–Inverse Document Frequency).
3. Each movie’s storyline is vectorized the same way.
4. **Cosine Similarity** compares user input with all movie storylines.
5. Top 5 matching movies are displayed as recommendations.

---

## 🧰 Technologies Used

- `Python 3`
- `Streamlit` (for interactive web app)
- `Selenium` (for web scraping IMDB)
- `Pandas` (for data cleaning and manipulation)
- `Scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
- `HTML/CSS` (for UI customization)

---

## 📦 Installation Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/adchayakumar/mini-project-05.git
   cd mini-project-05

## 🌐 Streamlit Application

- Input box to enter any storyline or plot.
- Filter movies by **rating**, **duration**, **genre**, and **title** using sidebar controls.
- Recommendations are shown in customized movie cards (built with HTML/CSS).
- **Lazy loading** enabled for filtered results using a "Show More" button.
- Simple, responsive, and user-friendly UI.

## To run the app locally:

```bash
streamlit run main.py

## Install dependencies:
  pip install -r requirements.txt
