import re
import numpy as np
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

# Initialize MobileNetV2 model for image feature extraction (CPU by default)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    # Use weights parameter for modern torchvision
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).to(device)
except Exception:
    # Fallback to pretrained=True for older versions
    mobilenet = models.mobilenet_v2(pretrained=True).to(device)
    
feature_extractor = mobilenet.features
feature_extractor.eval()

# Image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def clean_text(text):
    """
    Cleans title and description text by converting to lowercase, 
    removing duplicate spaces, newlines, and striping whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_ipq(content):
    """
    Extracts the Item Package Quantity (IPQ) from the product catalog content.
    Returns a float representation. Defaults to 1.0 if not found.
    """
    if not isinstance(content, str):
        return 1.0
    content_lower = content.lower()
    
    # 1. "pack of X"
    match = re.search(r'pack\s+of\s+(\d+)', content_lower)
    if match:
        return float(match.group(1))
    
    # 2. "pk of X" or "pkg of X"
    match = re.search(r'(?:pk|pkg)\s+of\s+(\d+)', content_lower)
    if match:
        return float(match.group(1))

    # 3. "pk- X" or "pack- X" or "pkg- X"
    match = re.search(r'(?:pk|pack|pkg)-\s*(\d+)', content_lower)
    if match:
        return float(match.group(1))

    # 4. "X pack" or "X pk" or "X count" or "X ct" or "X pcs" or "X pieces"
    match = re.search(r'\b(\d+)\s*(?:pack|pk|pkg|ct|count|pcs|pieces|bags|bottles|cans|tins|pouches|boxes|units|rolls|sets|pairs|sticks|servings|per case|per box)\b', content_lower)
    if match:
        return float(match.group(1))

    # 5. "X/case" or "X/pack" or "X/box" or "X/bag" or "X per case"
    match = re.search(r'\b(\d+)\s*(?:/|per\s+)(?:case|pack|box|bag|ct|pk)\b', content_lower)
    if match:
        return float(match.group(1))

    return 1.0

def get_image_features(image_source):
    """
    Extracts a 1280-dimensional feature vector from an image.
    image_source can be an image URL (string) or an opened PIL Image / file-like object.
    Returns a numpy array of shape (1280,) or None if extraction fails.
    """
    try:
        if isinstance(image_source, str) and image_source.startswith('http'):
            # Download image from URL
            response = requests.get(image_source, timeout=4)
            if response.status_code != 200:
                return None
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Load from file upload or PIL Image
            img = Image.open(image_source).convert('RGB')
            
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        
        with torch.no_grad():
            features = feature_extractor(batch_t)
            # Global average pooling
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features)
            return features.cpu().numpy()
    except Exception as e:
        print(f"Error extracting image features: {e}")
        return None

def extract_standardized_features(content):
    """
    Extracts the standardized product weight/volume or count value from the catalog content.
    Returns (standardized_value, is_count).
    """
    if not isinstance(content, str):
        return 1.0, 0.0
    
    value_match = re.search(r'Value:\s*([\d.]+)', content)
    unit_match = re.search(r'Unit:\s*([a-zA-Z\s]+)', content)
    
    value = float(value_match.group(1)) if value_match else 1.0
    unit = unit_match.group(1).lower().strip() if unit_match else "unknown"
    
    is_count = 0.0
    std_val = value
    
    if any(u in unit for u in ['count', 'ct', 'each', 'pack', 'piece', 'pcs', 'unit', 'none', 'bag', 'bottle', 'can', 'tin', 'box']):
        is_count = 1.0
        std_val = value
    elif any(u in unit for u in ['ounce', 'oz', 'fl', 'fluid']):
        std_val = value
    elif any(u in unit for u in ['pound', 'lb']):
        std_val = value * 16.0
    elif any(u in unit for u in ['gram', 'g', 'gm']):
        std_val = value / 28.35
    elif any(u in unit for u in ['millilitre', 'ml']):
        std_val = value / 29.57
    elif any(u in unit for u in ['liter', 'litre', 'l']):
        std_val = value * 33.81
    
    return std_val, is_count

def compute_seasonal_adjustment(content, season):
    """
    Computes seasonal/festival price adjustment multiplier and explanation.
    Returns (multiplier, explanation).
    """
    if not isinstance(content, str) or not season:
        return 1.0, "Normal Demand"
    
    content_lower = content.lower()
    
    if season == "festive":
        # Indian festival high demand items: Ghee, Masala, Spices, Basmati Rice, Sweets/Desserts
        if any(w in content_lower for w in ['ghee', 'masala', 'spice', 'basmati', 'rice', 'sweet', 'haldiram', 'dessert', 'curry', 'cardamom', 'saffron', 'coconut']):
            return 1.15, "High Festive Demand (Spices, Ghee, Basmati & Sweets)"
        return 1.05, "Moderate Festive Demand (General groceries)"
        
    elif season == "summer":
        # Summer items: Cold drinks, juices, salad dressings, popcorn/snacks, coconut
        if any(w in content_lower for w in ['juice', 'beverage', 'drink', 'coconut', 'salad', 'dressing', 'popcorn', 'snack', 'ice']):
            return 1.12, "Peak Summer Demand (Beverages, Dressings & Snacks)"
        return 0.95, "Slight Off-season Discount (Hot soups & spices)"
        
    elif season == "winter":
        # Winter items: Soups, hot tea, coffee, cardamom, spices, ghee
        if any(w in content_lower for w in ['soup', 'tea', 'coffee', 'chai', 'cardamom', 'ginger', 'spicy', 'ghee', 'hot']):
            return 1.15, "Peak Winter Demand (Soups, Hot Beverages & Warming Spices)"
        return 0.95, "Slight Off-season Discount (Summer cold beverages)"
        
    elif season == "monsoon":
        # Monsoon items: Hot tea/chai, soups, spicy foods/masalas
        if any(w in content_lower for w in ['tea', 'chai', 'soup', 'masala', 'spic', 'fry', 'oil']):
            return 1.10, "Monsoon Season Demand (Tea, Soups & Savory Spices)"
        return 1.0, "Normal Demand"
        
    return 1.0, "Normal Demand"
