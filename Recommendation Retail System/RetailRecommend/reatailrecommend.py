import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import joblib

# Set robust device detection for macOS (MPS) / NVIDIA (CUDA) / CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    use_cuda = False
    pin_memory = False
elif torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
    pin_memory = True
else:
    device = torch.device("cpu")
    use_cuda = False
    pin_memory = False

print(f"Using compute device: {device}")

# ==============================================================================
# 1. DATA PREPARATION WITH LOGQ POPULARITY CORRECTION
# ==============================================================================
def load_and_preprocess(events_path, props1_path, props2_path, tree_path):
    print("Loading CSV files...")
    events = pd.read_csv(events_path)
    props1 = pd.read_csv(props1_path)
    props2 = pd.read_csv(props2_path)
    tree_df = pd.read_csv(tree_path)

    props = pd.concat([props1, props2], ignore_index=True)
    del props1, props2
    gc.collect()

    valid_actions = ['view', 'addtocart', 'transaction']
    events = events[events['event'].isin(valid_actions)].dropna(subset=['itemid', 'visitorid']).copy()
    events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')

    # Category hierarchy
    cat_df = props[props['property'] == 'categoryid'][['itemid', 'value']].drop_duplicates('itemid')
    cat_df['categoryid'] = pd.to_numeric(cat_df['value'], errors='coerce').fillna(-1).astype(int)
    tree_df['parentid'] = pd.to_numeric(tree_df['parentid'], errors='coerce').fillna(tree_df['categoryid']).astype(int)
    cat_to_parent = dict(zip(tree_df['categoryid'], tree_df['parentid']))
    cat_df['parent_categoryid'] = cat_df['categoryid'].map(cat_to_parent).fillna(cat_df['categoryid']).astype(int)

    events = events.merge(cat_df[['itemid', 'categoryid', 'parent_categoryid']], on='itemid', how='left')
    events['categoryid'] = events['categoryid'].fillna(-1).astype(int)
    events['parent_categoryid'] = events['parent_categoryid'].fillna(-1).astype(int)

    del props, cat_df
    gc.collect()

    # Encoders
    u_enc, i_enc, c_enc, p_enc = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()
    events['u_idx'] = u_enc.fit_transform(events['visitorid'])
    events['i_idx'] = i_enc.fit_transform(events['itemid'])
    events['c_idx'] = c_enc.fit_transform(events['categoryid'])
    events['p_idx'] = p_enc.fit_transform(events['parent_categoryid'])

    weight_map = {'view': 1.0, 'addtocart': 2.5, 'transaction': 5.0}
    events['weight'] = events['event'].map(weight_map).astype(np.float32)

    # Chronological Split
    events = events.sort_values('timestamp').reset_index(drop=True)
    n = len(events)
    train_df = events.iloc[:int(n * 0.8)].copy()
    val_df   = events.iloc[int(n * 0.8):int(n * 0.9)].copy()
    test_df  = events.iloc[int(n * 0.9):].copy()

    train_items = set(train_df['i_idx'].unique())
    train_users = set(train_df['u_idx'].unique())
    val_df = val_df[val_df['i_idx'].isin(train_items) & val_df['u_idx'].isin(train_users)].reset_index(drop=True)
    test_df = test_df[test_df['i_idx'].isin(train_items) & test_df['u_idx'].isin(train_users)].reset_index(drop=True)

    item_meta = events[['i_idx', 'c_idx', 'p_idx']].drop_duplicates('i_idx').sort_values('i_idx')
    item_to_cat = torch.tensor(item_meta['c_idx'].values, dtype=torch.long)
    item_to_parent = torch.tensor(item_meta['p_idx'].values, dtype=torch.long)

    item_counts = train_df['i_idx'].value_counts().sort_index()
    num_items = len(i_enc.classes_)
    frequencies = np.ones(num_items, dtype=np.float32)
    frequencies[item_counts.index] += item_counts.values
    sampling_prob = frequencies / frequencies.sum()
    log_q = torch.tensor(np.log(sampling_prob + 1e-10), dtype=torch.float32)

    encoders = {'user': u_enc, 'item': i_enc, 'cat': c_enc, 'parent': p_enc}
    raw_dfs = {'events': events, 'tree': tree_df}

    return raw_dfs, train_df, val_df, test_df, encoders, item_to_cat, item_to_parent, log_q

# ==============================================================================
# 2. RESIDUAL ARCHITECTURE WITH STABLE NORMALIZATION
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.act(self.ln(self.fc(x))))

class OptimizedTwoTower(nn.Module):
    def __init__(self, num_users, num_items, num_cats, num_parents, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        self.user_proj = nn.Linear(emb_dim, emb_dim)
        self.user_res = ResidualBlock(emb_dim)

        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.cat_emb = nn.Embedding(num_cats, 16)
        self.parent_emb = nn.Embedding(num_parents, 16)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.cat_emb.weight, std=0.01)
        nn.init.normal_(self.parent_emb.weight, std=0.01)

        self.item_proj = nn.Linear(emb_dim + 16 + 16, emb_dim)
        self.item_res = ResidualBlock(emb_dim)

        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.07)))

    def forward_user(self, u_idx):
        u = self.user_emb(u_idx)
        x = self.user_proj(u)
        x = self.user_res(x)
        return F.normalize(x, p=2, dim=-1)

    def forward_item(self, i_idx, c_idx, p_idx):
        xi = self.item_emb(i_idx)
        xc = self.cat_emb(c_idx)
        xp = self.parent_emb(p_idx)
        x = torch.cat([xi, xc, xp], dim=-1)
        x = self.item_proj(x)
        x = self.item_res(x)
        return F.normalize(x, p=2, dim=-1)

    @property
    def temperature(self):
        return torch.exp(self.log_temp).clamp(min=0.02, max=0.2)

# ==============================================================================
# 3. TRAINING ROUTINE WITH GRADIENT ACCUMULATION
# ==============================================================================
class RecommenderDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.c = torch.tensor(df['c_idx'].values, dtype=torch.long)
        self.p = torch.tensor(df['p_idx'].values, dtype=torch.long)
        self.w = torch.tensor(df['weight'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.c[idx], self.p[idx], self.w[idx]

def logq_contrastive_loss(user_vecs, item_vecs, item_indices, weights, log_q_table, temp):
    sim_matrix = torch.matmul(user_vecs, item_vecs.T) / temp
    batch_log_q = log_q_table[item_indices].unsqueeze(0)
    sim_matrix = sim_matrix - batch_log_q
    labels = torch.arange(user_vecs.size(0), device=user_vecs.device)
    loss = F.cross_entropy(sim_matrix, labels, reduction='none')
    return (loss * weights).mean()

def train_model(model, train_loader, optimizer, scheduler, log_q_table, epochs=6, accum_steps=2):
    history = []
    log_q_dev = log_q_table.to(device)

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (u, i, c, p, w) in enumerate(train_loader):
            u = u.to(device, non_blocking=True)
            i = i.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)

            u_emb = model.forward_user(u)
            i_emb = model.forward_item(i, c, p)

            loss = logq_contrastive_loss(u_emb, i_emb, i, w, log_q_dev, model.temperature)
            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        lr_current = optimizer.param_groups[0]['lr']
        temp_val = float(model.temperature.item())
        history.append({
            "Epoch": ep,
            "Training Loss": round(epoch_loss, 5),
            "LR": round(lr_current, 6),
            "Temperature": round(temp_val, 4)
        })
        print(f"Epoch {ep:02d}/{epochs} - Loss: {epoch_loss:.4f} - LR: {lr_current:.6f} - Temp: {temp_val:.4f}")

    return history

# ==============================================================================
# 4. VECTORIZED RANKING EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate_and_generate_predictions(model, test_df, encoders, item_to_cat, item_to_parent, max_eval_samples=5000):
    model.eval()
    eval_df = test_df.head(max_eval_samples).copy()

    total_items = len(item_to_cat)
    all_items = torch.arange(total_items, device=device)
    all_cats = item_to_cat.to(device)
    all_parents = item_to_parent.to(device)

    # Embed items in chunks
    item_embs_list = []
    chunk_size = 16384
    for start_idx in range(0, total_items, chunk_size):
        end_idx = min(start_idx + chunk_size, total_items)
        emb_chunk = model.forward_item(
            all_items[start_idx:end_idx],
            all_cats[start_idx:end_idx],
            all_parents[start_idx:end_idx]
        )
        item_embs_list.append(emb_chunk.cpu())
    item_embs = torch.cat(item_embs_list, dim=0).to(device)

    inv_user = encoders['user'].inverse_transform
    inv_item = encoders['item'].inverse_transform

    user_tensor = torch.tensor(eval_df['u_idx'].values, dtype=torch.long, device=device)
    user_embs = model.forward_user(user_tensor)

    k_max = 20
    u_chunk = 512
    hits = {5: [], 10: [], 20: []}
    ndcgs = {5: [], 10: [], 20: []}
    mrrs = []
    predictions_rows = []
    all_recommended_items = set()

    for idx_start in range(0, len(eval_df), u_chunk):
        idx_end = min(idx_start + u_chunk, len(eval_df))
        u_batch = user_embs[idx_start:idx_end]
        target_batch = eval_df['i_idx'].values[idx_start:idx_end]

        sims = torch.matmul(u_batch, item_embs.T)
        top_scores, top_indices = torch.topk(sims, k=k_max, dim=-1)

        top_indices_np = top_indices.cpu().numpy()
        top_scores_np = top_scores.cpu().numpy()

        for b in range(len(target_batch)):
            tgt = target_batch[b]
            preds = top_indices_np[b]
            scs = top_scores_np[b]

            all_recommended_items.update(preds.tolist())

            if tgt in preds:
                rank = int(np.where(preds == tgt)[0][0]) + 1
                mrrs.append(1.0 / rank)
            else:
                rank = -1
                mrrs.append(0.0)

            for k in [5, 10, 20]:
                k_preds = preds[:k]
                hit_k = int(tgt in k_preds)
                hits[k].append(hit_k)
                if hit_k:
                    r_k = int(np.where(k_preds == tgt)[0][0]) + 1
                    ndcgs[k].append(1.0 / np.log2(r_k + 1))
                else:
                    ndcgs[k].append(0.0)

            if len(predictions_rows) < 5000:
                row_data = eval_df.iloc[idx_start + b]
                raw_actual = inv_item([tgt])[0]
                raw_preds = inv_item(preds).tolist()
                predictions_rows.append({
                    'visitorid': inv_user([row_data.u_idx])[0],
                    'actual_itemid': raw_actual,
                    'event_type': row_data.event,
                    'hit@10': hits[10][-1],
                    'hit@20': hits[20][-1],
                    'hit_rank': rank,
                    'mrr': round(mrrs[-1], 4),
                    'ndcg@10': round(ndcgs[10][-1], 4),
                    'ndcg@20': round(ndcgs[20][-1], 4),
                    'top_1_itemid': raw_preds[0],
                    'top_1_sim_score': round(float(scs[0]), 4),
                    'top_5_items': str(raw_preds[:5]),
                    'top_10_items': str(raw_preds[:10]),
                    'top_20_items': str(raw_preds[:20])
                })

    preds_df = pd.DataFrame(predictions_rows)

    metrics_summary = pd.DataFrame({
        'Evaluation Metric': [
            'Hit@5', 'NDCG@5', 'Hit@10', 'NDCG@10', 'Hit@20', 'NDCG@20',
            'MRR (Mean Reciprocal Rank)', 'Catalog Coverage (%)',
            'Evaluated Samples', 'Total Candidate Catalog Size'
        ],
        'Score / Value': [
            round(float(np.mean(hits[5])), 4),
            round(float(np.mean(ndcgs[5])), 4),
            round(float(np.mean(hits[10])), 4),
            round(float(np.mean(ndcgs[10])), 4),
            round(float(np.mean(hits[20])), 4),
            round(float(np.mean(ndcgs[20])), 4),
            round(float(np.mean(mrrs)), 4),
            round((len(all_recommended_items) / total_items) * 100, 2),
            len(eval_df),
            total_items
        ]
    })

    event_segments = preds_df.groupby('event_type').agg(
        Sample_Count=('actual_itemid', 'count'),
        Hit_at_10=('hit@10', 'mean'),
        NDCG_at_10=('ndcg@10', 'mean'),
        Hit_at_20=('hit@20', 'mean'),
        MRR=('mrr', 'mean')
    ).round(4).reset_index()

    return preds_df, metrics_summary, event_segments

# ==============================================================================
# 5. CONSOLIDATED SINGLE-FILE EXCEL WRITER
# ==============================================================================
def write_master_excel_report(raw_dfs, train_df, val_df, test_df, training_hist, metrics_df, event_segments, preds_df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Writing all sheets into consolidated workbook: {output_path}")

    events = raw_dfs['events']

    # Overview Table
    overview_df = pd.DataFrame({
        'Attribute': ['Total Events', 'Unique Visitors', 'Unique Items', 'Unique Categories', 'Unique Parent Categories'],
        'Value': [len(events), events['visitorid'].nunique(), events['itemid'].nunique(), events['categoryid'].nunique(), events['parent_categoryid'].nunique()]
    })

    # Event Actions
    event_counts = events['event'].value_counts().reset_index()
    event_counts.columns = ['Event Action', 'Total Count']
    event_counts['Percentage'] = (event_counts['Total Count'] / len(events) * 100).round(2).astype(str) + '%'

    # Split Distribution
    split_df = pd.DataFrame({
        'Partition': ['Train (80%)', 'Validation (10%)', 'Test (10%)'],
        'Event Count': [len(train_df), len(val_df), len(test_df)]
    })

    # Model Hyperparameters
    config_df = pd.DataFrame({
        'Hyperparameter': ['Device', 'Embedding Dim', 'Batch Size (Physical)', 'Accumulation Steps', 'Effective Batch Size', 'Loss'],
        'Value': [str(device), '64', '1024', '2', '2048', 'LogQ In-Batch Contrastive']
    })

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: High Level EDA
        overview_df.to_excel(writer, sheet_name='EDA_Overview', index=False, startrow=0)
        event_counts.to_excel(writer, sheet_name='EDA_Overview', index=False, startrow=len(overview_df) + 3)
        split_df.to_excel(writer, sheet_name='EDA_Overview', index=False, startrow=len(overview_df) + len(event_counts) + 6)

        # Sheet 2: Training Epoch Loss Log & Config
        pd.DataFrame(training_hist).to_excel(writer, sheet_name='Training_History', index=False, startrow=0)
        config_df.to_excel(writer, sheet_name='Training_History', index=False, startrow=len(training_hist) + 3)

        # Sheet 3: Global Ranking KPIs & Metrics
        metrics_df.to_excel(writer, sheet_name='Model_KPIs', index=False)

        # Sheet 4: Segment Breakdown (views vs carts vs purchases)
        event_segments.to_excel(writer, sheet_name='Segment_Accuracy', index=False)

        # Sheet 5: User-level Actuals vs Top-K Predictions
        preds_df.to_excel(writer, sheet_name='Actuals_vs_Predictions', index=False)

    print(f"Consolidated Excel report complete with 5 sheets.")

# ==============================================================================
# 6. PIPELINE ORCHESTRATOR
# ==============================================================================
def run_pipeline():
    # Detect RetailRecommend script folder and project root dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Configured data and output directories
    data_dir = os.path.join(project_root, "Data")
    output_dir = os.path.join(project_root, "Output")
    os.makedirs(output_dir, exist_ok=True)

    master_excel_path = os.path.join(output_dir, "Output/recommendation_system_master_report.xlsx")

    events_path = os.path.join(data_dir, "events.csv")
    props1_path = os.path.join(data_dir, "item_properties_part1.csv")
    props2_path = os.path.join(data_dir, "item_properties_part2.csv")
    tree_path   = os.path.join(data_dir, "category_tree.csv")

    for p in [events_path, props1_path, props2_path, tree_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    raw_dfs, train_df, val_df, test_df, encoders, item_to_cat, item_to_parent, log_q = load_and_preprocess(
        events_path, props1_path, props2_path, tree_path
    )

    train_loader = DataLoader(
        RecommenderDataset(train_df),
        batch_size=1024,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=0
    )

    num_users = len(encoders['user'].classes_)
    num_items = len(encoders['item'].classes_)
    num_cats = len(encoders['cat'].classes_)
    num_parents = len(encoders['parent'].classes_)

    model = OptimizedTwoTower(num_users, num_items, num_cats, num_parents, emb_dim=64).to(device)

    sparse_params = [model.user_emb.weight, model.item_emb.weight, model.cat_emb.weight, model.parent_emb.weight]
    dense_params = [p for p in model.parameters() if not any(p is sp for sp in sparse_params)]

    optimizer = torch.optim.AdamW([
        {'params': sparse_params, 'lr': 2e-3, 'weight_decay': 1e-6},
        {'params': dense_params, 'lr': 1e-3, 'weight_decay': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    print("Beginning Training Routine (6 Epochs with Memory Optimization)...")
    training_history = train_model(model, train_loader, optimizer, scheduler, log_q, epochs=10, accum_steps=2)

    print("Evaluating Test Set Predictions...")
    preds_df, metrics_df, event_segments = evaluate_and_generate_predictions(
        model, test_df, encoders, item_to_cat, item_to_parent, max_eval_samples=5000
    )

    # Write all data to single Excel file
    write_master_excel_report(
        raw_dfs, train_df, val_df, test_df, training_history, metrics_df, event_segments, preds_df,
        output_path=master_excel_path
    )

    # Save artifacts
    artifacts_dir = os.path.join(output_dir, "Recommendation Retail System/Output")
    os.makedirs(artifacts_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(artifacts_dir, "optimized_two_tower.pth"))
    for name, enc in encoders.items():
        joblib.dump(enc, os.path.join(artifacts_dir, f"{name}_encoder.pkl"))
    print(f"Artifacts saved in {artifacts_dir}/")

if __name__ == "__main__":
    run_pipeline()