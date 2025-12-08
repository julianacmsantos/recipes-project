# backend/app/main.py
#
# Camada de API da aplicação.
# Responsável por:
# - inicializar o FastAPI
# - configurar CORS e logging
# - carregar o motor de recomendação na inicialização
# - expor os endpoints /health e /recommend

import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .model_utils import RecipeRecommender

# -----------------------------------------------------------------------------
# Configuração básica de logging
# Define o formato e o nível de log
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
# Log específico para a API
logger = logging.getLogger("recipes-api")

# -----------------------------------------------------------------------------
# Carrega variáveis de ambiente e caminhos de arquivos
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# .env ficará na raiz do backend (ex: backend/.env) ou no projeto
# Permite configurar caminhos e origens sem alterar o código
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

DEFAULT_INDEX_PATH = os.path.join(BASE_DIR, "..", "embeddings_index", "faiss_index.index")
DEFAULT_META_PATH = os.path.join(BASE_DIR, "..", "embeddings_index", "metadata.csv")

INDEX_PATH = os.getenv("INDEX_PATH", DEFAULT_INDEX_PATH)
META_PATH = os.getenv("META_PATH", DEFAULT_META_PATH)

FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")

# -----------------------------------------------------------------------------
# Inicialização da aplicação FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Recipe Recommender API",
    description="API para recomendação de receitas a partir de ingredientes, usando embeddings + FAISS.",
    version="0.1.0",
)

# CORS – necessário para o frontend (React) acessar a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in FRONTEND_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instância global do recomendador (inicializada no startup)
recommender: Optional[RecipeRecommender] = None


# -----------------------------------------------------------------------------
# Schemas Pydantic (modelos de entrada/saída da API)
# -----------------------------------------------------------------------------
class Query(BaseModel):
    """
    Modelo de requisição para o endpoint /recommend.

    Exemplo de JSON recebido:
    {
      "ingredients": "tomato garlic olive oil",
      "top_k": 5
    }
    """
    ingredients: str = Field(..., description="Lista de ingredientes em texto livre.")
    top_k: int = Field(5, ge=1, le=50, description="Quantidade máxima de receitas retornadas.")


class Recipe(BaseModel):
    """
    Representa uma receita individual retornada pela recomendação.

    Os campos devem refletir as colunas presentes em metadata.csv,
    além dos campos calculados similarity_score e match_percent.
    """
    id: int
    title: str
    ingredients: str
    instructions: Optional[str] = None
    link: Optional[str] = None
    ner: Optional[str] = None
    similarity_score: float = Field(..., ge=-1.0, le=1.0)
    match_percent: float = Field(..., ge=0.0, le=100.0)


class RecommendResponse(BaseModel):
    """
    Modelo de resposta do endpoint /recommend.

    Exemplo de JSON:
    {
      "query": "tomato garlic olive oil",
      "results": [ { ...Recipe... }, ... ]
    }
    """
    query: str
    results: List[Recipe]
    message: Optional[str] = Field(None, description="Nenhuma receita encontrada.")


# -----------------------------------------------------------------------------
# Eventos da aplicação
# -----------------------------------------------------------------------------
@app.on_event("startup")
def startup_event() -> None:
    """
    Carrega o sistema de recomendação na inicialização da API.

    Responsável por:
    - carregar a metadata das receitas
    - carregar o modelo de embeddings
    - carregar o índice FAISS

    Isso evita carregar o modelo/índice durante o import do módulo.
    """
    global recommender

    logger.info("Iniciando carregamento do sistema de recomendação...")
    try:
        recommender = RecipeRecommender(INDEX_PATH, META_PATH) #
        logger.info("Sistema carregado e pronto para uso!")
    except Exception as exc:
        logger.exception("Falha ao inicializar o RecipeRecommender: %s", exc)
        # Deixa o recommender como None; os endpoints tratarão isso
        recommender = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Endpoint simples de health check.
    """
    status = "ok" if recommender is not None else "degraded"
    message = "API está viva!" if recommender is not None else "API viva, mas recomendador não inicializado."
    return {"status": status, "message": message}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(q: Query):
    """
    Endpoint principal de recomendação de receitas por ingredientes.
    """
    if recommender is None:
        logger.error("Tentativa de uso do /recommend sem recommender inicializado.")
        raise HTTPException(
            status_code=503,
            detail="Sistema de recomendação não está disponível no momento. Verifique os logs do servidor.",
        )

    user_text = q.ingredients.strip().lower()
    logger.info("Nova consulta recebida: '%s' | top_k=%d", user_text, q.top_k)

    if not user_text:
        logger.warning("Consulta vazia recebida no /recommend.")
        raise HTTPException(
            status_code=400,
            detail="O campo 'ingredients' não pode ser vazio.",
        )

    try:
        results = recommender.recommend(
            user_input=user_text,
            top_k=q.top_k,
            require_exact_token=True
        )
        
        if not results:
            logger.info("Nenhuma receita encontrada para a consulta do usuário.")
            return RecommendResponse(
                query=user_text,results=[],
                message="Nenhuma receita encontrada para os ingredientes fornecidos."
            )

        logger.info("Retornando %d resultados para o usuário.", len(results))

        return RecommendResponse(query=user_text, results=results)

    except Exception as exc:
        logger.exception("Erro ao processar recomendação: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Ocorreu um erro ao processar a recomendação.",
        ) from exc
