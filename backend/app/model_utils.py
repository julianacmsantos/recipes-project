# backend/app/model_utils.py

import logging

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("recipes-api.recommender")


class RecipeRecommender:
    """
    Wrapper simples responsável por:
      - carregar metadata das receitas;
      - carregar o modelo de embeddings;
      - carregar o índice FAISS;
      - expor um método de recomendação.
    """

    def __init__(
        self,
        index_path: str,
        meta_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        logger.info("Carregando metadata das receitas de '%s'...", meta_path)
        self.meta = pd.read_csv(meta_path)
        logger.info("Metadata carregada com %d receitas.", len(self.meta))

        logger.info("Carregando modelo de embeddings '%s'...", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Modelo de embeddings carregado com sucesso.")

        logger.info("Lendo índice FAISS do caminho '%s'...", index_path)
        self.index = faiss.read_index(index_path)
        self.dim = self.index.d
        logger.info("Índice FAISS carregado com %d vetores (dim=%d).", self.index.ntotal, self.dim)

        logger.info("RecipeRecommender inicializado com sucesso.\n")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Gera o embedding normalizado (L2) para o texto informado.
        """
        logger.debug("Gerando embedding para o texto do usuário.")
        emb = self.model.encode([text], convert_to_numpy=True)
        emb = emb.astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def recommend(self, user_input: str, top_k: int = 10):
        """
        Retorna uma lista de receitas mais similares ao texto de entrada.
        Cada item é um dicionário com os campos da metadata + scores.
        """
        emb = self.embed_text(user_input)

        logger.info("Buscando os %d resultados mais similares no índice FAISS...", top_k)
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

        logger.info("Busca finalizada com %d resultados.\n", len(results))
        return results
