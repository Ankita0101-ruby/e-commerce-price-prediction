import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import feature helpers
from features import clean_text, extract_ipq, get_image_features, extract_standardized_features

print("--- Smart Product Pricing Prediction Training Pipeline ---")

# Paths
csv_path = "d:/e-commerce price/Final_project.csv"
# Robust path checking for cached features
cached_features_path = "C:/Users/ankit/.gemini/antigravity/brain/3d222089-1023-41bc-9aa5-8d4bce20d0d5/scratch/multimodal_features.pkl"
if not os.path.exists(cached_features_path):
    cached_features_path = "C:/Users/ankit/.gemini/antigravity/brain/ff662f59-8bcf-4611-b1cb-505470ea2d95/scratch/multimodal_features.pkl"

model_assets_path = "d:/e-commerce price/model_assets.pkl"

print("Loading dataset...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows.")

# 1. PREPROCESSING TEXT & IPQ
print("Preprocessing text and extracting IPQ...")
df['clean_text'] = df['catalog_content'].apply(clean_text)
df['ipq'] = df['catalog_content'].apply(extract_ipq)
df['ipq_capped'] = np.clip(df['ipq'], 1.0, 100.0)

# Extract standardized features
print("Extracting standardized features...")
std_features = df['catalog_content'].apply(extract_standardized_features)
df['std_value'] = [f[0] for f in std_features]
df['is_count'] = [f[1] for f in std_features]
df['std_value_capped'] = np.clip(df['std_value'], 0.01, 1000.0)

# Target price values (already capped in CSV, but let's confirm)
y_full = df['price']

# 2. TRAIN FULL TEXT + IPQ MODEL (XGBoost)
print("\n--- Training Full Text + IPQ Model (XGBoost) ---")
# Split train/test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

print("Vectorizing full text...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_text = vectorizer.fit_transform(train_df['clean_text'])
X_test_text = vectorizer.transform(test_df['clean_text'])

# Combine numerical features (IPQ + std_value_capped + is_count)
X_train_num = np.column_stack([
    train_df['ipq_capped'].values,
    train_df['std_value_capped'].values,
    train_df['is_count'].values
])
X_test_num = np.column_stack([
    test_df['ipq_capped'].values,
    test_df['std_value_capped'].values,
    test_df['is_count'].values
])

X_train_full = hstack([X_train_text, csr_matrix(X_train_num)])
X_test_full = hstack([X_test_text, csr_matrix(X_test_num)])

print("Training XGBoost Regressor on full dataset...")
start_time = time.time()
# Fine-tuned parameters for good generalization and speed
xgb_model = xgb.XGBRegressor(
    n_estimators=150, 
    max_depth=6, 
    learning_rate=0.08, 
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1, 
    random_state=42
)
xgb_model.fit(X_train_full, train_df['price'])
print(f"XGBoost training took {time.time() - start_time:.2f} seconds.")

# Evaluate full XGBoost model
y_pred_xgb = xgb_model.predict(X_test_full)
print(f"Full XGBoost R2 Score: {r2_score(test_df['price'], y_pred_xgb):.4f}")
print(f"Full XGBoost MAE: {mean_absolute_error(test_df['price'], y_pred_xgb):.4f}")

# 3. TRAIN MULTIMODAL MODEL (Text + IPQ + Images)
print("\n--- Loading Multimodal Features ---")
multimodal_data = None

# Try loading cached features
if os.path.exists(cached_features_path):
    try:
        with open(cached_features_path, 'rb') as f:
            multimodal_data = pickle.load(f)
        print(f"Loaded {len(multimodal_data)} cached image features.")
    except Exception as e:
        print("Error reading cached features, extracting new ones:", e)

# If no cached features, extract on the fly for a subset of 1000 items
if multimodal_data is None:
    print("No cached multimodal features found. Extracting on the fly for 1000 items (this may take 2-3 mins)...")
    sample_sub = df.sample(n=1000, random_state=42)
    multimodal_data = []
    
    # Simple extraction loop
    extracted_count = 0
    for idx, row in sample_sub.iterrows():
        features = get_image_features(row['image_link'])
        if features is not None:
            multimodal_data.append({
                'sample_id': row['sample_id'],
                'image_features': features,
                'clean_text': row['clean_text'],
                'ipq': row['ipq_capped'],
                'price': row['price']
            })
            extracted_count += 1
            if extracted_count % 100 == 0:
                print(f"Extracted features for {extracted_count} items...")
                
    print(f"Extracted features for {len(multimodal_data)} items.")

# Process multimodal dataset
mm_df = pd.DataFrame(multimodal_data)

# Merge with the main df to get std_value_capped and is_count
mm_df = mm_df.merge(df[['sample_id', 'std_value_capped', 'is_count']], on='sample_id', how='left')

mm_train, mm_test = train_test_split(mm_df, test_size=0.15, random_state=42)

# Vectorize text on multimodal subset (use full vectorizer transforms to keep vocabulary consistent)
X_mm_train_text = vectorizer.transform(mm_train['clean_text'])
X_mm_test_text = vectorizer.transform(mm_test['clean_text'])

# Combine numerical features (IPQ + std_value_capped + is_count)
X_mm_train_num = np.column_stack([
    mm_train['ipq'].values,
    mm_train['std_value_capped'].values,
    mm_train['is_count'].values
])
X_mm_test_num = np.column_stack([
    mm_test['ipq'].values,
    mm_test['std_value_capped'].values,
    mm_test['is_count'].values
])

X_mm_train_text_ipq = hstack([X_mm_train_text, csr_matrix(X_mm_train_num)])
X_mm_test_text_ipq = hstack([X_mm_test_text, csr_matrix(X_mm_test_num)])

# Extract and scale image features
X_mm_train_img = np.vstack(mm_train['image_features'].values)
X_mm_test_img = np.vstack(mm_test['image_features'].values)

scaler = StandardScaler()
X_mm_train_img_scaled = scaler.fit_transform(X_mm_train_img)
X_mm_test_img_scaled = scaler.transform(X_mm_test_img)

X_mm_train_all = hstack([X_mm_train_text_ipq, csr_matrix(X_mm_train_img_scaled)])
X_mm_test_all = hstack([X_mm_test_text_ipq, csr_matrix(X_mm_test_img_scaled)])

print("\nTraining Multimodal Ridge Regressor...")
mm_model = Ridge(alpha=5000.0)
mm_model.fit(X_mm_train_all, mm_train['price'])

y_pred_mm = mm_model.predict(X_mm_test_all)
print(f"Multimodal Ridge R2 Score: {r2_score(mm_test['price'], y_pred_mm):.4f}")
print(f"Multimodal Ridge MAE: {mean_absolute_error(mm_test['price'], y_pred_mm):.4f}")

# 4. SAVE PRODUCTION ASSETS
print(f"\nSaving model assets to {model_assets_path}...")
assets = {
    'vectorizer': vectorizer,
    'scaler': scaler,
    'xgb_model': xgb_model,
    'mm_model': mm_model
}

with open(model_assets_path, 'wb') as f:
    pickle.dump(assets, f)

print("Training pipeline completed and assets saved successfully!")
