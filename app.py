import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from scipy.sparse import hstack, csr_matrix

# Import feature extraction functions
from features import clean_text, extract_ipq, get_image_features, extract_standardized_features, compute_seasonal_adjustment

app = Flask(__name__)

# Paths
MODEL_ASSETS_PATH = "d:/e-commerce price/model_assets.pkl"
DATABASE_PATH = "d:/e-commerce price/Final_project.csv"

# Global variables to hold models and data
models = None
db_df = None

def load_assets():
    global models, db_df
    
    # Load model assets
    if os.path.exists(MODEL_ASSETS_PATH):
        print("Loading model assets...")
        with open(MODEL_ASSETS_PATH, 'rb') as f:
            models = pickle.load(f)
        print("Model assets loaded successfully.")
    else:
        print("WARNING: Model assets not found yet! Predictions will fail until assets are trained.")

    # Load dataset for search and browse
    if os.path.exists(DATABASE_PATH):
        print("Loading product database...")
        # Only load necessary columns to keep memory low
        db_df = pd.read_csv(DATABASE_PATH, usecols=['sample_id', 'catalog_content', 'image_link', 'price'])
        db_df = db_df.fillna('')
        print(f"Database loaded with {len(db_df)} records.")
    else:
        print("WARNING: Product database CSV not found!")

# Load assets on startup
load_assets()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/random-samples', methods=['GET'])
def get_random_samples():
    if db_df is None:
        return jsonify({'error': 'Database not loaded'}), 500
        
    samples = db_df.sample(n=5).to_dict(orient='records')
    return jsonify(samples)

@app.route('/api/search', methods=['GET'])
def search_products():
    if db_df is None:
        return jsonify({'error': 'Database not loaded'}), 500
        
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
        
    # Search in catalog_content (case insensitive)
    matches = db_df[db_df['catalog_content'].str.lower().str.contains(query, na=False)]
    
    # Return top 12 matches for UI responsiveness
    results = matches.head(12).to_dict(orient='records')
    return jsonify(results)

@app.route('/api/predict', methods=['POST'])
def predict_price():
    global models
    if models is None:
        # Retry loading once in case they were trained after start
        load_assets()
        if models is None:
            return jsonify({'error': 'Model assets not trained yet'}), 500

    # Get inputs
    title_desc = request.form.get('title_desc', '')
    quantity = request.form.get('quantity', '')
    image_url = request.form.get('image_url', '')
    image_file = request.files.get('image_file')
    season = request.form.get('season', 'normal')

    # Preprocess inputs
    text_cleaned = clean_text(title_desc)
    
    # Handle quantity
    try:
        ipq = float(quantity) if quantity else extract_ipq(title_desc)
    except ValueError:
        ipq = extract_ipq(title_desc)
    
    # Cap IPQ
    ipq_capped = np.clip(ipq, 1.0, 100.0)

    # Extract standardized features
    std_val, is_count = extract_standardized_features(title_desc)
    std_val_capped = np.clip(std_val, 0.01, 1000.0)

    # Transform text using loaded vectorizer
    X_text = models['vectorizer'].transform([text_cleaned])
    
    # Combine numerical features
    num_features = np.array([[ipq_capped, std_val_capped, is_count]])
    X_num = csr_matrix(num_features)
    X_text_ipq = hstack([X_text, X_num])

    # Determine image features
    img_features = None
    prediction_type = "Text + Quantity Model (XGBoost)"
    
    # Check for image file upload first (explicit custom upload)
    if image_file and image_file.filename:
        print("Extracting features from uploaded image file...")
        img_features = get_image_features(image_file)
        prediction_type = "Multimodal Model (Text + Quantity + Uploaded Image)"
    elif image_url and image_url.strip().startswith('http'):
        # For database items or URL inputs, we prefer using the highly accurate full XGBoost model.
        prediction_type = "High-Accuracy Text + Quantity Model (XGBoost)"

    predicted_price = 0.0

    # Choose model based on image features availability
    if img_features is not None:
        try:
            # Scale image features
            img_features_scaled = models['scaler'].transform([img_features])
            
            # Combine all features
            X_all = hstack([X_text_ipq, csr_matrix(img_features_scaled)])
            
            # Predict
            pred = models['mm_model'].predict(X_all)[0]
            # Ensure price is non-negative
            predicted_price = max(0.01, float(pred))
        except Exception as e:
            print(f"Multimodal inference failed: {e}. Falling back to Text+IPQ.")
            img_features = None  # Force fallback
            
    if img_features is None:
        # Fallback to full XGBoost model
        if "XGBoost" not in prediction_type:
            prediction_type = "Text + Quantity Model (XGBoost)"
        pred = models['xgb_model'].predict(X_text_ipq)[0]
        predicted_price = max(0.01, float(pred))

    # Apply seasonal adjustment
    multiplier, seasonal_desc = compute_seasonal_adjustment(title_desc, season)
    adjusted_price = predicted_price * multiplier

    # Return results
    return jsonify({
        'predicted_price': round(adjusted_price, 2),
        'base_price': round(predicted_price, 2),
        'multiplier': multiplier,
        'seasonal_explanation': seasonal_desc,
        'model_used': prediction_type,
        'features': {
            'text_length': len(title_desc),
            'extracted_quantity': ipq_capped,
            'has_image': img_features is not None
        }
    })
@app.route('/api/chat', methods=['POST'])
def chat():
    global models, db_df
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'response': "Please enter a message."})
        
    msg_lower = message.lower()
    
    # Help / Info Intent
    if any(k in msg_lower for k in ['help', 'how to', 'how does', 'what can you do', 'menu']):
        return jsonify({
            'response': (
                "🤖 **NEXORA AI Assistant at your service!** Here is how I can help:\n\n"
                "1. **Search the Catalog**: Type `search ghee` or `find MDH` to search our database of 75,000 products.\n"
                "2. **Predict Price**: Type `predict Everest Masala` or enter a product title, and I will estimate its price.\n"
                "3. **Random Suggestion**: Type `suggest a product` or `random` to get a sample from the catalog.\n"
                "4. **Model Info**: Ask me how predictions are made, and I'll explain the XGBoost and Multimodal features."
            )
        })
        
    # Model Explanation Intent
    if any(k in msg_lower for k in ['model', 'xgboost', 'prediction work', 'explain price', 'how do you predict']):
        return jsonify({
            'response': (
                "📈 **How NEXORA Predicts Prices:**\n\n"
                "- **Text Feature Extraction**: We analyze keywords, brand, and size info using TF-IDF vectorization.\n"
                "- **IPQ (Items Per Quantity)**: Our engine automatically extracts pack sizes (e.g. 'Pack of 6') from descriptions.\n"
                "- **Multimodal Integration**: When an image is provided, ResNet features are merged with text representations.\n"
                "- **Seasonal Adjustments**: Prices are adjusted dynamically during festive or summer seasons using demand modifiers."
            )
        })

    # Suggestion Intent
    if any(k in msg_lower for k in ['suggest', 'random', 'recommend', 'shuffle']):
        if db_df is None:
            return jsonify({'response': "Database is not loaded."})
        sample = db_df.sample(n=1).iloc[0]
        # Clean title
        title = sample['catalog_content'].split('\n')[0].replace('Item Name:', '').strip()
        return jsonify({
            'response': f"🎲 **Product Recommendation:**\n\n**{title}**\n*Actual Catalog Price:* `${float(sample['price']):.2f}`",
            'product': {
                'sample_id': int(sample['sample_id']),
                'catalog_content': sample['catalog_content'],
                'image_link': sample['image_link'],
                'price': float(sample['price'])
            }
        })

    # Search Intent
    search_keywords = ['search for', 'find', 'search', 'look up', 'show me']
    query = ""
    for kw in search_keywords:
        if msg_lower.startswith(kw):
            query = message[len(kw):].strip()
            break
            
    # Fallback to direct search if it looks like a generic query for specific products
    if not query and any(k in msg_lower for k in ['ghee', 'masala', 'mustard', 'oil', 'dabur', 'everest', 'mdh', 'rice', 'basmati', 'shampoo', 'soap', 'neutrogena', 'vaseline', 'cetaphil', 'nivea', 'olay', 'loreal', 'dove', 'lakme', 'clinic', 'garnier', 'ponds', 'mamaearth', 'mama earth', 'fortune', 'freedom', 'dairymilk', 'dairy milk', 'kitkat', 'kit kat', 'hersheys', 'hershey', 'lays', 'kurkure', 'gooday', 'good day', 'unibic', 'indiagate', 'india gate', 'ruchi']):
        # If it doesn't contain 'predict', default to search
        if 'predict' not in msg_lower:
            query = message.strip()
            
    if query:
        if db_df is None:
            return jsonify({'response': "Database is not loaded."})
        
        matches = db_df[db_df['catalog_content'].str.lower().str.contains(query.lower(), na=False)]
        if matches.empty:
            return jsonify({
                'response': f"🔍 I searched the catalog for **\"{query}\"** but couldn't find any exact matches. Try searching for other terms like 'ghee', 'masala', or 'dabur'."
            })
            
        results = matches.head(3).to_dict(orient='records')
        
        response_text = f"🔍 **Search Results for \"{query}\":**\n"
        products_list = []
        for item in results:
            title = item['catalog_content'].split('\n')[0].replace('Item Name:', '').strip()
            if len(title) > 60:
                title = title[:57] + "..."
            response_text += f"\n- **{title}** (${float(item['price']):.2f})"
            products_list.append({
                'sample_id': int(item['sample_id']),
                'catalog_content': item['catalog_content'],
                'image_link': item['image_link'],
                'price': float(item['price'])
            })
            
        return jsonify({
            'response': response_text,
            'products': products_list
        })
        
    # Prediction Intent
    predict_keywords = ['predict price of', 'predict price for', 'predict', 'estimate', 'how much is']
    pred_query = ""
    for kw in predict_keywords:
        if msg_lower.startswith(kw):
            pred_query = message[len(kw):].strip()
            break
            
    if pred_query:
        if models is None:
            return jsonify({'response': "Prediction model is not loaded."})
            
        # Try to find a match in DB first
        matched_item = None
        if db_df is not None:
            matches = db_df[db_df['catalog_content'].str.lower().str.contains(pred_query.lower(), na=False)]
            if not matches.empty:
                matched_item = matches.iloc[0]
                
        if matched_item is not None:
            title_desc = matched_item['catalog_content']
            actual_price = float(matched_item['price'])
        else:
            title_desc = pred_query
            actual_price = None
            
        # Run prediction
        text_cleaned = clean_text(title_desc)
        ipq = extract_ipq(title_desc)
        ipq_capped = np.clip(ipq, 1.0, 100.0)
        std_val, is_count = extract_standardized_features(title_desc)
        std_val_capped = np.clip(std_val, 0.01, 1000.0)
        
        X_text = models['vectorizer'].transform([text_cleaned])
        num_features = np.array([[ipq_capped, std_val_capped, is_count]])
        X_num = csr_matrix(num_features)
        X_text_ipq = hstack([X_text, X_num])
        
        pred = models['xgb_model'].predict(X_text_ipq)[0]
        predicted_price = max(0.01, float(pred))
        
        title_display = title_desc.split('\n')[0].replace('Item Name:', '').strip()
        if len(title_display) > 60:
            title_display = title_display[:57] + "..."
            
        response_msg = f"🔮 **NEXORA Prediction Engine** calculated standard price:\n\n"
        response_msg += f"**Product:** *{title_display}*\n"
        response_msg += f"**Predicted Price:** `${predicted_price:.2f}`\n"
        if actual_price is not None:
            response_msg += f"**Actual Catalog Price:** `${actual_price:.2f}`"
            
        product_data = None
        if matched_item is not None:
            product_data = {
                'sample_id': int(matched_item['sample_id']),
                'catalog_content': matched_item['catalog_content'],
                'image_link': matched_item['image_link'],
                'price': float(matched_item['price'])
            }
            
        return jsonify({
            'response': response_msg,
            'product': product_data
        })
        
    # Default Fallback (unknown intent)
    return jsonify({
        'response': (
            "👋 Hello! I am the **NEXORA AI Chatbot**. I can help you search the catalog or predict prices.\n\n"
            "Try one of these suggestions:\n"
            "• `suggest a product` to get a sample\n"
            "• `search for basmati rice` to find it in the database\n"
            "• `predict Everest Masala` to run pricing prediction"
        )
    })

import urllib.request
import urllib.parse
import json

@app.route('/api/translate', methods=['POST'])
def translate_text():
    data = request.get_json() or {}
    text = data.get('text', '')
    target_lang = data.get('target', 'en')
    source_lang = data.get('source', 'auto')
    
    if not text:
        return jsonify({'translated_text': ''})
        
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            res_json = json.loads(response.read().decode('utf-8'))
            translated_parts = [part[0] for part in res_json[0] if part[0]]
            translated_text = "".join(translated_parts)
            return jsonify({'translated_text': translated_text})
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'translated_text': text, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
