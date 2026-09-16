"""
End-to-End News Recommendation System Pipeline
=================================================
Pipeline Workflow:
1. Dynamic Dataset Discovery & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Latent Semantic Analysis & User Interaction Modeling
4. Rule-Based, Content-Based, and Hybrid Algorithms
5. Offline Performance Benchmarking (Precision@K, Recall@K, MAP@K, NDCG@K, MRR)
6. Multi-Sheet Excel Report Export via openpyxl / pd.ExcelWriter:
   - 'EDA_Summary'
   - 'EDA_Keyword_Stats'
   - 'Model_Evaluation_Metrics'
   - 'Actuals_vs_Predicted'
   - 'Top_K_Recommendations'
"""

import ast
from datetime import datetime
import os
from pathlib import Path
import re
import numpy as np
import openpyxl
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# 1. LOAD & PREPROCESS DATA
# =====================================================================
print("[Step 1/6] Loading and Preprocessing Data...")

# Dynamically find 'result_final.csv' across probable directory structures
current_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
search_locations = [
    current_dir / "Recomendation News Model/Dataset/result_final.csv",
    current_dir.parent / "Recomendation News Model/Dataset/result_final.csv",
    current_dir.parents[1] / "Recomendation News Model/Dataset/result_final.csv" if len(current_dir.parents) > 1 else current_dir,
    Path.cwd() / "Recomendation News Model/Dataset/result_final.csv",
    Path.cwd() / "Dataset" / "result_final.csv",
]

csv_file_path = None
for loc in search_locations:
    if loc.exists() and loc.is_file():
        csv_file_path = loc
        break

if not csv_file_path:
    raise FileNotFoundError(
        "Could not find 'result_final.csv'. Please place it in the same directory or check your working path."
    )

print(f"Reading dataset from: {csv_file_path}")
df = pd.read_csv(csv_file_path)

# Drop any auto-generated index columns
drop_cols = [c for c in ["Unnamed: 0.1", "Unnamed: 0"] if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)

# Fill null values for text processing
df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)
df["summary"] = df["summary"].fillna("").astype(str)
df["keywords"] = df["keywords"].fillna("[]").astype(str)

# Assign a deterministic article ID
df = df.reset_index(drop=True)
df["article_id"] = df.index


def parse_keywords(kw_str):
    """Safely extracts a list of lowercase string tokens from keyword representations."""
    try:
        val = ast.literal_eval(kw_str)
        if isinstance(val, list):
            return [w.strip().lower() for w in val if isinstance(w, str)]
    except Exception:
        pass
    cleaned = re.sub(r"[\[\]\'\"]", "", str(kw_str))
    return [w.strip().lower() for w in cleaned.split(",") if w.strip()]


df["parsed_keywords"] = df["keywords"].apply(parse_keywords)
df["keyword_count"] = df["parsed_keywords"].apply(len)
df["text_word_count"] = df["text"].apply(lambda x: len(x.split()))
df["summary_word_count"] = df["summary"].apply(lambda x: len(x.split()))
df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")

# Handle missing dates using the median timestamp
median_date = df["parsed_date"].dropna().median()
df["clean_date"] = df["parsed_date"].fillna(
    median_date if pd.notnull(median_date) else datetime(2020, 1, 1)
)

# Enriched content field: Title weighted twice + Summary + Keywords
df["content_features"] = (
    df["title"]
    + " "
    + df["title"]
    + " "
    + df["summary"]
    + " "
    + df["parsed_keywords"].apply(lambda x: " ".join(x))
)

# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================
print("[Step 2/6] Performing EDA Analysis...")

eda_summary = pd.DataFrame(
    [
        {"Metric": "Total Articles", "Value": len(df)},
        {"Metric": "Unique Titles", "Value": df["title"].replace("", np.nan).nunique()},
        {"Metric": "Missing Date Records", "Value": int(df["date"].isnull().sum())},
        {"Metric": "Mean Text Word Count", "Value": round(df["text_word_count"].mean(), 2)},
        {"Metric": "Median Text Word Count", "Value": round(df["text_word_count"].median(), 2)},
        {"Metric": "Mean Summary Word Count", "Value": round(df["summary_word_count"].mean(), 2)},
        {"Metric": "Earliest Article Date", "Value": str(df["clean_date"].min())},
        {"Metric": "Latest Article Date", "Value": str(df["clean_date"].max())},
    ]
)

all_keywords = [kw for kw_list in df["parsed_keywords"] for kw in kw_list if len(kw) > 1]

# Fully compatible with both Pandas 1.x and Pandas 2.x/3.x
kw_series = pd.Series(all_keywords).value_counts().head(50)
keyword_freq = kw_series.reset_index()
keyword_freq.columns = ["Keyword", "Frequency"]
kw_popularity_dict = keyword_freq.set_index("Keyword")["Frequency"].to_dict()

# =====================================================================
# 3. SEMANTIC VECTORIZATION & INTERACTION SIMULATION
# =====================================================================
print("[Step 3/6] Vectorization and User Interaction Simulation...")

tfidf = TfidfVectorizer(max_features=4000, stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content_features"])

lsa = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = lsa.fit_transform(tfidf_matrix)

# Simulate 60 user reading patterns with 70% train history and 30% holdout ground truth
np.random.seed(42)
NUM_USERS = 60
user_profiles = []

for u in range(NUM_USERS):
    seed_item = np.random.randint(0, len(df))
    sims = cosine_similarity(latent_matrix[seed_item : seed_item + 1], latent_matrix).ravel()
    candidate_items = np.argsort(-sims)[1:35]

    chosen = np.random.choice(candidate_items, size=min(12, len(candidate_items)), replace=False)
    split_idx = int(len(chosen) * 0.7)
    train_history = list(chosen[:split_idx])
    test_actuals = list(chosen[split_idx:])

    user_profiles.append(
        {
            "user_id": f"User_{u+1:03d}",
            "train_items": train_history,
            "test_items": test_actuals,
        }
    )

# =====================================================================
# 4. RECOMMENDATION ALGORITHMS
# =====================================================================
print("[Step 4/6] Building Recommender Functions...")

# Precompute Recency Signal
min_time = df["clean_date"].astype("int64").min()
max_time = df["clean_date"].astype("int64").max()
time_diff = max_time - min_time if max_time != min_time else 1.0
df["recency_score"] = (df["clean_date"].astype("int64") - min_time) / time_diff


# Precompute Global Keyword Prominence Signal
def compute_keyword_score(kw_list):
    if not kw_list:
        return 0.0
    return sum(kw_popularity_dict.get(k, 0) for k in kw_list) / len(kw_list)


df["keyword_pop_score"] = df["parsed_keywords"].apply(compute_keyword_score)
max_pop = df["keyword_pop_score"].max()
df["keyword_pop_score"] = df["keyword_pop_score"] / (max_pop if max_pop > 0 else 1.0)


def recommend_rule_based(history_ids, top_k=10):
    """Rule-Based: 40% Publication Recency + 40% Keyword Frequency + 20% Direct Overlap."""
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
    scores.loc[history_ids] = -1.0  # Exclude already read items
    top_indices = scores.nlargest(top_k).index.tolist()
    return top_indices, scores


def recommend_content_based(history_ids, top_k=10):
    """Content-Based: Average semantic cosine similarity over the user's history."""
    user_vector = latent_matrix[history_ids].mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(user_vector, latent_matrix).flatten()
    scores = pd.Series(sims, index=df.index)
    scores.loc[history_ids] = -1.0
    top_indices = scores.nlargest(top_k).index.tolist()
    return top_indices, scores


def recommend_hybrid(history_ids, top_k=10, alpha=0.65):
    """Hybrid: Combines 65% Latent Semantic Similarity with 35% Rule-Based signals."""
    _, rule_scores = recommend_rule_based(history_ids, top_k=top_k)
    _, cb_scores = recommend_content_based(history_ids, top_k=top_k)

    rule_norm = (rule_scores - rule_scores.min()) / (
        rule_scores.max() - rule_scores.min() + 1e-9
    )
    cb_norm = (cb_scores - cb_scores.min()) / (
        cb_scores.max() - cb_scores.min() + 1e-9
    )

    hybrid_scores = (alpha * cb_norm) + ((1.0 - alpha) * rule_norm)
    hybrid_scores.loc[history_ids] = -1.0
    top_indices = hybrid_scores.nlargest(top_k).index.tolist()
    return top_indices, hybrid_scores


# =====================================================================
# 5. OFFLINE BENCHMARK EVALUATION
# =====================================================================
print("[Step 5/6] Benchmarking Models & Computing Metrics...")


def compute_metrics(actual_items, rec_items, k=10):
    rec_k = rec_items[:k]
    hits = [1 if item in actual_items else 0 for item in rec_k]
    num_hits = sum(hits)

    prec = num_hits / k
    recall = num_hits / len(actual_items) if len(actual_items) > 0 else 0.0

    # Mean Average Precision (MAP@K)
    score = 0.0
    num_rel = 0
    for i, h in enumerate(rec_k):
        if h == 1:
            num_rel += 1
            score += num_rel / (i + 1.0)
    map_k = score / min(len(actual_items), k) if actual_items else 0.0

    # Normalized Discounted Cumulative Gain (NDCG@K)
    dcg = sum((2**h - 1) / np.log2(idx + 2) for idx, h in enumerate(hits))
    ideal_hits = [1] * min(len(actual_items), k) + [0] * max(0, k - len(actual_items))
    idcg = sum((2**h - 1) / np.log2(idx + 2) for idx, h in enumerate(ideal_hits))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    # Mean Reciprocal Rank (MRR)
    first_hit = next((i + 1 for i, h in enumerate(rec_k) if h == 1), 0)
    rr = 1.0 / first_hit if first_hit > 0 else 0.0

    return {
        "Precision@K": prec,
        "Recall@K": recall,
        "MAP@K": map_k,
        "NDCG@K": ndcg,
        "MRR": rr,
    }


K = 10
models = {
    "Rule_Based": recommend_rule_based,
    "Content_Based": recommend_content_based,
    "Hybrid_Best_Model": recommend_hybrid,
}

eval_rows = []
actuals_vs_predicted_rows = []
recommendation_outputs = []

for u_data in user_profiles:
    uid = u_data["user_id"]
    train = u_data["train_items"]
    test = u_data["test_items"]

    actual_titles = [df.loc[i, "title"] for i in test]

    for m_name, model_fn in models.items():
        recs, _ = model_fn(train, top_k=K)
        m_eval = compute_metrics(test, recs, k=K)

        eval_rows.append({"Model": m_name, "User_ID": uid, **m_eval})

        rec_titles = [df.loc[i, "title"] for i in recs]
        matched_hits = list(set(test).intersection(set(recs)))

        actuals_vs_predicted_rows.append(
            {
                "User_ID": uid,
                "Model": m_name,
                "Actual_Item_IDs": str(test),
                "Actual_Count": len(test),
                "Predicted_Item_IDs": str(recs),
                "Hits_Count": len(matched_hits),
                "Hit_Item_IDs": str(matched_hits),
                "Precision@10": round(m_eval["Precision@K"], 4),
                "Recall@10": round(m_eval["Recall@K"], 4),
                "NDCG@10": round(m_eval["NDCG@K"], 4),
                "Actual_Sample_Title": actual_titles[0] if actual_titles else "N/A",
                "Top1_Predicted_Title": rec_titles[0] if rec_titles else "N/A",
            }
        )

        if m_name == "Hybrid_Best_Model":
            for rank, item_id in enumerate(recs, 1):
                recommendation_outputs.append(
                    {
                        "User_ID": uid,
                        "Rank": rank,
                        "Recommended_Article_ID": item_id,
                        "Recommended_Title": df.loc[item_id, "title"],
                        "Article_Date": str(df.loc[item_id, "date"]),
                        "Keywords": ", ".join(df.loc[item_id, "parsed_keywords"][:6]),
                        "Is_Ground_Truth_Match": "YES" if item_id in test else "NO",
                    }
                )

df_eval = pd.DataFrame(eval_rows)
metrics_summary = (
    df_eval.groupby("Model")[["Precision@K", "Recall@K", "MAP@K", "NDCG@K", "MRR"]]
    .mean()
    .reset_index()
)
metrics_summary = metrics_summary.sort_values(by="NDCG@K", ascending=False).reset_index(drop=True)

df_act_pred = pd.DataFrame(actuals_vs_predicted_rows)
df_top_k_recs = pd.DataFrame(recommendation_outputs)

# =====================================================================
# 6. WRITE TO EXCEL WRITER (Multi-Sheet Workbook)
# =====================================================================
output_excel =  "Recommendation Models/Recomendation News Model/Output/recommendation_system_results.xlsx"
print(f"[Step 6/6] Writing results to Excel: {output_excel}...")

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    eda_summary.to_excel(writer, sheet_name="EDA_Summary", index=False)
    keyword_freq.to_excel(writer, sheet_name="EDA_Keyword_Stats", index=False)
    metrics_summary.to_excel(writer, sheet_name="Model_Evaluation_Metrics", index=False)
    df_act_pred.to_excel(writer, sheet_name="Actuals_vs_Predicted", index=False)
    df_top_k_recs.head(1000).to_excel(writer, sheet_name="Top_K_Recommendations", index=False)

print(f"\nPipeline successfully completed! Results written to: {output_excel}")
print("\nModel Benchmark Summary:")
print(metrics_summary.to_string(index=False))