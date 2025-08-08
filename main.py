import os, re
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
from sentence_transformers import SentenceTransformer, util
import logging

app = FastAPI()

# Enable CORS (consider limiting origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load products safely
try:
    with open("products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
except Exception as e:
    logging.error(f"Failed to load products.json: {e}")
    products = []

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create product embeddings once
if products:
    product_texts = [
        f"{p['name']} {p['description']} category: {p['category']} price: {p['price']} rating: {p['rating']}"
        for p in products
    ]
    product_embeddings = model.encode(product_texts, convert_to_tensor=True)
else:
    product_embeddings = None

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
import re


def parse_query_for_filters(query: str):
    price_max, price_min, min_rating = None, None, None

    # Look for "less than X" or "under X" price
    match = re.search(r'(?:less than|less|under)\s*\$?(\d+)', query, re.IGNORECASE)
    if match:
        price_max = float(match.group(1))

    # Look for "more than X" or "over X" price
    match = re.search(r'(?:more than|more|upper|over)\s*\$?(\d+)', query, re.IGNORECASE)
    if match:
        price_min = float(match.group(1))

    # Look for "rating above X" or "rating greater than X"
    match = re.search(r'(?:rating\s*(?:above|greater than|over))\s*(\d+(\.\d+)?)', query, re.IGNORECASE)
    if match:
        min_rating = float(match.group(1))

    return price_min, price_max, min_rating


@app.get("/")
async def read_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Index file not found")


@app.get("/search")
def smart_search(query: str = Query(..., description="Search query"), category: str = None):
    if not products or product_embeddings is None:
        raise HTTPException(status_code=500, detail="Product data not loaded")

    # Extract numeric constraints from query
    price_min, price_max, min_rating = parse_query_for_filters(query)

    try:
        query_embedding = model.encode(query, convert_to_tensor=True)
    except Exception as e:
        logging.error(f"Failed to encode query: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query")

    scores = util.cos_sim(query_embedding, product_embeddings)[0]

    # Create result list
    results = []
    for i, prod in enumerate(products):
        # Apply numeric filters
        if price_min is not None and prod['price'] < price_min:
            continue
        if price_max is not None and prod['price'] > price_max:
            continue
        if min_rating is not None and prod['rating'] < min_rating:
            continue

        # Apply category filter (optional)
        if category and prod['category'].lower() != category.lower():
            continue

        results.append({**prod, "score": float(scores[i])})

    # Sort by semantic score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:5]
