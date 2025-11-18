# backend/app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import os

from model_utils import RecipeRecommender

app = FastAPI(title="Recipe Recommender API")

# Caminhos até os arquivos gerados no Colab
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "embeddings_index", "faiss_index.index")
META_PATH  = os.path.join(BASE_DIR, "..", "embeddings_index", "metadata.csv")

print("🛠 Iniciando carregamento do sistema de recomendação...")
recommender = RecipeRecommender(INDEX_PATH, META_PATH)
print("🌟 Sistema carregado e pronto para uso!")

class Query(BaseModel):
    ingredients: str
    top_k: int = 10

@app.get("/health")
def health():
    return {"status": "ok", "message": "API está viva!"}

@app.post("/recommend")
def recommend(q: Query):
    print("📨 Recebi uma nova consulta do usuário.")
    user_text = q.ingredients.lower()
    print(f"🔎 Texto processado: {user_text}")

    results = recommender.recommend(user_text, top_k=q.top_k)

    print("📤 Enviando resultados ao usuário.\n")
    return {
        "query": user_text,
        "results": results
    }
