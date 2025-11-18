# backend/app/model_utils.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import os

class RecipeRecommender:
    def __init__(self, index_path, meta_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print("🔄 Carregando metadata das receitas...")
        self.meta = pd.read_csv(meta_path)
        print(f"📄 Metadata carregada com {len(self.meta)} receitas.")

        print("🧠 Carregando modelo de embeddings...")
        self.model = SentenceTransformer(model_name)
        print("✨ Modelo carregado: ok.")

        print("📦 Lendo índice FAISS do disco...")
        self.index = faiss.read_index(index_path)
        self.dim = self.index.d
        print(f"📚 Índice carregado com {self.index.ntotal} vetores embeddados.")

        print("🚀 Recommender inicializado com sucesso.\n")

    def embed_text(self, text):
        print("📝 Gerando embedding da consulta do usuário...")
        emb = self.model.encode([text], convert_to_numpy=True)
        emb = emb.astype('float32')
        faiss.normalize_L2(emb)
        return emb

    def recommend(self, user_input, top_k=10):
        emb = self.embed_text(user_input)

        print(f"🔍 Buscando resultados mais parecidos (top {top_k})...")
        D, I = self.index.search(emb, top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue

            row = self.meta.iloc[idx].to_dict()

            # Conversão simples de similaridade → porcentagem
            percent = max(0.0, float((score + 1) / 2 * 100.0))

            row["similarity_score"] = float(score)
            row["match_percent"] = round(percent, 2)

            results.append(row)

        print("✅ Busca finalizada.\n")
        return results
