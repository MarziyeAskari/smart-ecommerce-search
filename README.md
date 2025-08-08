# smart-ecommerce-search
A simple semantic search engine for an e-commerce product catalog built with FastAPI and Sentence Transformers.
---

## Features

- **Semantic Search**: Uses SentenceTransformers' `"all-MiniLM-L6-v2"` model to encode product descriptions and user queries, allowing search by meaning rather than just keywords.
- **Numeric Filtering**: Supports filtering products based on price and rating parsed from user queries.
- **Category Filtering**: Optionally filter search results by product category.
- **Frontend UI**: Simple interactive webpage with live search and category dropdown.
- **API**: FastAPI backend serving search requests and static frontend.

---

## Tools & Libraries

- **FastAPI** — for creating the REST API and serving static files.
- **SentenceTransformers** — to compute semantic embeddings of text.
- **Uvicorn** — ASGI server to run the FastAPI app.
- **JavaScript (Fetch API)** — frontend communication with backend API.
- **JSON** — product catalog data.

---

## How to Run the App

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MarziyeAskari/smart-ecommerce-search.git
   cd smart-ecommerce-search
