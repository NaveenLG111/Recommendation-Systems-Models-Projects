"""
End-to-End Streamlit Web Application for News Recommendation System
-------------------------------------------------------------------
Features:
1. Article & History Recommender:
   - Select multiple seed articles consumed by a reader (user profile).
   - Dynamically run Rule-Based, Content-Based, or Hybrid ranking algorithms.
   - Adjust Hybrid ensemble weighting (Alpha parameter) with real-time updates.
2. Similar News Finder:
   - Item-to-item latent semantic similarity search for related articles.
3. Offline Model Benchmark & Excel Dashboard:
   - Interactive viewer for all sheets in 'recommendation_system_results.xlsx'
     ('EDA_Summary', 'EDA_Keyword_Stats', 'Model_Evaluation_Metrics',
      'Actuals_vs_Predicted', 'Top_K_Recommendations').
   - Download the generated multi-sheet Excel report directly from the UI.
"""

import ast
from datetime import datetime
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="News Recommendation Engine",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. PATH RESOLUTION & DATA INGESTION
# ==============================================================================
@st.cache_resource(show_spinner=True)
def resolve_file_paths():
    script_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    
    # Candidate locations to inspect
    search_paths = [
        # Explicit full paths
        Path("/Users/naveenlg/AI Models/Recommendation Models/Recomendation News Model/Dataset/result_final.csv"),
        Path("/Users/naveenlg/AI Models/Recommendation Models/Recomendation News Model/Newsfiles/result_final.csv"),
        Path("/Users/naveenlg/AI Models/result_final.csv"),
        # Relative to script location
        script_dir / "result_final.csv",
        script_dir / "Dataset" / "result_final.csv",
        script_dir.parent / "Dataset" / "result_final.csv",
        script_dir.parents[1] / "Dataset" / "result_final.csv",
        # Relative to current working directory
        Path.cwd() / "result_final.csv",
        Path.cwd() / "Dataset" / "result_final.csv",
        Path.cwd() / "Recomendation News Model" / "Dataset" / "result_final.csv",
        Path.cwd() / "Recommendation Models" / "Recomendation News Model" / "Dataset" / "result_final.csv",
    ]

    csv_path = next((p for p in search_paths if p.exists() and p.is_file()), None)

    # Search for the output Excel file
    excel_search = [
        Path("/Users/naveenlg/AI Models/Recommendation Models/Recomendation News Model/Output/recommendation_system_results.xlsx"),
        script_dir.parent / "Output" / "recommendation_system_results.xlsx",
        script_dir / "recommendation_system_results.xlsx",
        Path.cwd() / "recommendation_system_results.xlsx",
    ]
    excel_path = next((p for p in excel_search if p.exists() and p.is_file()), None)

    return csv_path, excel_path


def parse_keywords(kw_str):
    try:
        val = ast.literal_eval(kw_str)
        if isinstance(val, list):
            return [w.strip().lower() for w in val if isinstance(w, str)]
    except Exception:
        pass
    cleaned = re.sub(r"[\[\]\'\"]", "", str(kw_str))
    return [w.strip().lower() for w in cleaned.split(",") if w.strip()]


@st.cache_data(show_spinner="Preprocessing News Dataset & Vectorizing Text...")
def load_and_prepare_news_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop unnecessary index columns if present
    drop_cols = [c for c in ["Unnamed: 0.1", "Unnamed: 0"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["keywords"] = df["keywords"].fillna("[]").astype(str)

    df = df.reset_index(drop=True)
    df["article_id"] = df.index
    df["parsed_keywords"] = df["keywords"].apply(parse_keywords)

    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    median_date = df["parsed_date"].dropna().median()
    df["clean_date"] = df["parsed_date"].fillna(
        median_date if pd.notnull(median_date) else datetime(2020, 1, 1)
    )

    # Recency normalization (0 to 1)
    min_time = df["clean_date"].astype("int64").min()
    max_time = df["clean_date"].astype("int64").max()
    time_diff = max_time - min_time if max_time != min_time else 1.0
    df["recency_score"] = (df["clean_date"].astype("int64") - min_time) / time_diff

    # Keyword frequency dictionary
    all_keywords = [kw for kw_list in df["parsed_keywords"] for kw in kw_list if len(kw) > 1]
    kw_series = pd.Series(all_keywords).value_counts().head(100)
    kw_popularity_dict = kw_series.to_dict()

    def compute_keyword_score(kw_list):
        if not kw_list:
            return 0.0
        return sum(kw_popularity_dict.get(k, 0) for k in kw_list) / len(kw_list)

    df["keyword_pop_score"] = df["parsed_keywords"].apply(compute_keyword_score)
    max_pop = df["keyword_pop_score"].max()
    df["keyword_pop_score"] = df["keyword_pop_score"] / (max_pop if max_pop > 0 else 1.0)

    # Content Vectorization: Title (boosted 2x) + Summary + Keywords
    df["content_features"] = (
        df["title"]
        + " "
        + df["title"]
        + " "
        + df["summary"]
        + " "
        + df["parsed_keywords"].apply(lambda x: " ".join(x))
    )

    tfidf = TfidfVectorizer(max_features=4000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["content_features"])

    lsa = TruncatedSVD(n_components=50, random_state=42)
    latent_matrix = lsa.fit_transform(tfidf_matrix)

    return df, latent_matrix, kw_popularity_dict


# ==============================================================================
# 3. CORE RECOMMENDATION ALGORITHMS
# ==============================================================================
def run_rule_based(df, history_ids, kw_popularity_dict, top_k=10):
    user_kw = set(
        [k for idx in history_ids for k in df.loc[idx, "parsed_keywords"] if k in kw_popularity_dict]
    )

    def overlap_score(kws):
        return len(set(kws).intersection(user_kw))

    overlap = df["parsed_keywords"].apply(overlap_score)
    norm_overlap = overlap / (overlap.max() if overlap.max() > 0 else 1.0)

    scores = (
        0.40 * df["recency_score"] + 0.40 * df["keyword_pop_score"] + 0.20 * norm_overlap
    ).copy()
    scores.loc[history_ids] = -1.0
    return scores.nlargest(top_k).index.tolist(), scores


def run_content_based(df, latent_matrix, history_ids, top_k=10):
    user_vector = latent_matrix[history_ids].mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(user_vector, latent_matrix).flatten()
    scores = pd.Series(sims, index=df.index)
    scores.loc[history_ids] = -1.0
    return scores.nlargest(top_k).index.tolist(), scores


def run_hybrid(df, latent_matrix, history_ids, kw_popularity_dict, top_k=10, alpha=0.65):
    _, rule_scores = run_rule_based(df, history_ids, kw_popularity_dict, top_k=top_k)
    _, cb_scores = run_content_based(df, latent_matrix, history_ids, top_k=top_k)

    rule_norm = (rule_scores - rule_scores.min()) / (
        rule_scores.max() - rule_scores.min() + 1e-9
    )
    cb_norm = (cb_scores - cb_scores.min()) / (
        cb_scores.max() - cb_scores.min() + 1e-9
    )

    hybrid_scores = (alpha * cb_norm) + ((1.0 - alpha) * rule_norm)
    hybrid_scores.loc[history_ids] = -1.0
    return hybrid_scores.nlargest(top_k).index.tolist(), hybrid_scores


# ==============================================================================
# 4. MAIN APP INTERFACE
# ==============================================================================
st.title("📰 Intelligent News Recommendation System")
st.caption("Real-Time Multi-Strategy Recommendation Engine: Rule-Based, Content LSA, and Hybrid Ensemble")

csv_path, excel_path = resolve_file_paths()

if not csv_path:
    st.error("⚠️ Could not locate `result_final.csv`. Ensure the dataset exists in your project workspace.")
    st.stop()

df, latent_matrix, kw_popularity_dict = load_and_prepare_news_data(csv_path)

# Sidebar Configuration
st.sidebar.header("Navigation")
view_mode = st.sidebar.radio(
    "Select App View",
    ["Personalized Recommender", "Similar News Explorer", "Offline Benchmark & Excel Sheets"],
)

st.sidebar.divider()
st.sidebar.markdown(f"**Total Articles:** `{len(df):,}`")
st.sidebar.markdown(f"**Dataset Location:** `{csv_path.name}`")

# ==============================================================================
# VIEW 1: PERSONALIZED NEWS RECOMMENDER
# ==============================================================================
if view_mode == "Personalized Recommender":
    st.subheader("🎯 Reader Profile & Live Recommendations")
    st.write("Select one or more news articles to represent a reader's recent reading history.")

    col1, col2 = st.columns([2.5, 1])

    article_options = df["article_id"].tolist()

    with col1:
        selected_article_ids = st.multiselect(
            "Select Consumed Articles (Reading History):",
            options=article_options,
            default=[0, 1] if len(df) > 1 else [0],
            format_func=lambda x: f"ID {x}: {df.loc[x, 'title'][:85]}...",
        )

    with col2:
        model_choice = st.selectbox(
            "Recommendation Strategy:",
            ["Hybrid (Content + Rules)", "Content-Based (TF-IDF + LSA)", "Rule-Based (Recency + Prominence)"],
        )
        top_k = st.slider("Number of Recommendations (Top-K):", min_value=5, max_value=25, value=10, step=5)

    alpha_val = 0.65
    if "Hybrid" in model_choice:
        alpha_val = st.slider(
            "Hybrid Weight (Alpha): Content vs Rule Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
            help="Higher Alpha favors semantic content similarity; lower Alpha favors publication recency and popular keywords.",
        )

    if not selected_article_ids:
        st.warning("Please select at least one article in the reading history to generate personalized recommendations.")
        st.stop()

    # Display Reading History Preview
    with st.expander(f"📜 View Selected Reading History ({len(selected_article_ids)} Articles)", expanded=False):
        history_table = df.loc[selected_article_ids, ["article_id", "title", "date", "parsed_keywords"]].copy()
        history_table["parsed_keywords"] = history_table["parsed_keywords"].apply(lambda k: ", ".join(k[:6]))
        st.dataframe(history_table, use_container_width=True, hide_index=True)

    # Compute Recommendations
    if "Hybrid" in model_choice:
        rec_ids, score_series = run_hybrid(
            df, latent_matrix, selected_article_ids, kw_popularity_dict, top_k=top_k, alpha=alpha_val
        )
    elif "Content-Based" in model_choice:
        rec_ids, score_series = run_content_based(
            df, latent_matrix, selected_article_ids, top_k=top_k
        )
    else:
        rec_ids, score_series = run_rule_based(
            df, selected_article_ids, kw_popularity_dict, top_k=top_k
        )

    st.markdown(f"### 🚀 Recommended News Articles (Strategy: `{model_choice}`)")

    rec_results = []
    for rank, rid in enumerate(rec_ids, 1):
        rec_results.append(
            {
                "Rank": rank,
                "Article ID": rid,
                "Title": df.loc[rid, "title"],
                "Model Score": round(float(score_series.loc[rid]), 4),
                "Publication Date": str(df.loc[rid, "date"]),
                "Summary Preview": df.loc[rid, "summary"][:160] + "..." if df.loc[rid, "summary"] else "N/A",
                "Keywords": ", ".join(df.loc[rid, "parsed_keywords"][:6]),
            }
        )

    rec_df = pd.DataFrame(rec_results)
    st.dataframe(
        rec_df[["Rank", "Article ID", "Title", "Model Score", "Publication Date", "Keywords"]],
        use_container_width=True,
        hide_index=True,
    )

    # Detailed expandable cards for top 3 recommendations
    st.markdown("#### 🔍 Top 3 Detailed Summaries")
    top_3_cols = st.columns(min(3, len(rec_ids)))
    for idx, col in enumerate(top_3_cols):
        r_id = rec_ids[idx]
        with col:
            st.info(f"**#{idx+1}: {df.loc[r_id, 'title']}**")
            st.caption(f"📅 Date: {df.loc[r_id, 'date']} | 🏷️ Score: {round(float(score_series.loc[r_id]), 4)}")
            st.write(df.loc[r_id, "summary"][:280] + ("..." if len(df.loc[r_id, "summary"]) > 280 else ""))

# ==============================================================================
# VIEW 2: SIMILAR NEWS EXPLORER
# ==============================================================================
elif view_mode == "Similar News Explorer":
    st.subheader("🔍 Article-to-Article Semantic Similarity")
    st.write("Query an article to explore nearest neighbor articles in the 50-dimensional latent semantic space.")

    target_id = st.selectbox(
        "Select Target Article:",
        options=df["article_id"].tolist(),
        index=0,
        format_func=lambda x: f"ID {x}: {df.loc[x, 'title'][:90]}...",
    )
    sim_k = st.slider("Number of Similar Articles:", 5, 20, 10, 5)

    target_vec = latent_matrix[target_id : target_id + 1]
    cos_sims = cosine_similarity(target_vec, latent_matrix).flatten()
    cos_sims[target_id] = -1.0  # Exclude self

    top_similar_indices = np.argsort(-cos_sims)[:sim_k]

    sim_table = []
    for rank, s_idx in enumerate(top_similar_indices, 1):
        sim_table.append(
            {
                "Rank": rank,
                "Article ID": s_idx,
                "Title": df.loc[s_idx, "title"],
                "Cosine Similarity": round(float(cos_sims[s_idx]), 4),
                "Publication Date": str(df.loc[s_idx, "date"]),
                "Keywords": ", ".join(df.loc[s_idx, "parsed_keywords"][:6]),
            }
        )

    st.markdown(f"#### Articles Most Similar to: *{df.loc[target_id, 'title']}*")
    st.dataframe(pd.DataFrame(sim_table), use_container_width=True, hide_index=True)

# ==============================================================================
# VIEW 3: OFFLINE BENCHMARK & EXCEL WORKBOOK VIEWER
# ==============================================================================
elif view_mode == "Offline Benchmark & Excel Sheets":
    st.subheader("📊 Offline Benchmark Reports & Excel Workbook Viewer")

    if not excel_path or not excel_path.exists():
        st.warning(
            "⚠️ The results file `recommendation_system_results.xlsx` was not found. "
            "Please run your `newsrecommed.py` script first to generate the report."
        )
    else:
        st.success(f"Loaded Excel Report: `{excel_path.name}`")
        xls = pd.ExcelFile(excel_path)
        sheet_choice = st.selectbox("Select Excel Sheet to Inspect:", xls.sheet_names)

        sheet_df = pd.read_excel(excel_path, sheet_name=sheet_choice)
        st.dataframe(sheet_df, use_container_width=True)

        # Highlight KPI Cards if on Model_Evaluation_Metrics
        if sheet_choice == "Model_Evaluation_Metrics":
            st.divider()
            st.markdown("### 🏆 Algorithm Leaderboard Comparison")
            st.dataframe(
                sheet_df.style.highlight_max(
                    subset=[c for c in sheet_df.columns if c != "Model"],
                    color="#d1e7dd"
                ),
                use_container_width=True,
                hide_index=True,
            )

        with open(excel_path, "rb") as f:
            st.download_button(
                label="📥 Download Full recommendation_system_results.xlsx",
                data=f,
                file_name="Recommendation Models/Recomendation News Model/Output/recommendation_system_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )