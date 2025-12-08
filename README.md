# Recipes Project — Backend (FastAPI + FAISS + Embeddings)

Este é o backend do *Recipes Project*, um sistema de recomendação de receitas baseado em texto livre.
A API recebe uma lista de ingredientes e retorna as receitas mais similares, utilizando:

* Sentence Transformers para geração de embeddings semânticos
* FAISS para busca vetorial eficiente
* FastAPI como camada HTTP leve e moderna

O objetivo é oferecer um serviço rápido, estável e fácil de integrar ao frontend React/TypeScript.

---

## Tecnologias Utilizadas

* **FastAPI**
* **Sentence Transformers (MiniLM-L6-v2)**
* **FAISS (CPU)**
* **Python 3.10+**
* **Pandas e NumPy**
* **python-dotenv** para configuração
* **logging** para observabilidade

---

## Estrutura do Projeto

```
backend/
├── app/
│   ├── main.py                 # API, rotas, validação, CORS e carregamento do modelo
│   ├── model_utils.py          # Motor de recomendação (embeddings + FAISS)
│   └── ...
│
├── embeddings_index/
│   ├── faiss_index.index       # Índice FAISS pré-processado
│   ├── metadata.csv            # Metadata extraída do RecipeNLG (100k receitas)
│
├── requirements.txt
└── README.md
```

---

## Como Executar o Backend

### 1. Clonar o repositório

```bash
git clone https://github.com/julianacmsantos/recipes-project.git
cd recipes-project/backend
```

### 2. Criar e ativar um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3. Instalar dependências

```bash
pip install -r app/requirements.txt
```

### 4. Criar o arquivo `.env`

Crie `backend/.env` com:

```
INDEX_PATH=app/../embeddings_index/faiss_index.index
META_PATH=app/../embeddings_index/metadata.csv
FRONTEND_ORIGINS=http://localhost:5173,http://localhost:3000
```

### 5. Iniciar o servidor

```bash
uvicorn app.main:app --reload
```

A API estará disponível em:

```
http://localhost:8000
```

---

## Endpoints Disponíveis

### 1. Health Check

```
GET /health
```

Verifica se a API está no ar e se o mecanismo de recomendação foi carregado corretamente.

### 2. Recomendação de Receitas

```
POST /recommend
```

Corpo esperado:

```json
{
  "ingredients": "tomato garlic olive oil",
  "top_k": 5
}
```

Exemplo de resposta:

```json
{
  "query": "tomato garlic olive oil",
  "results": [
    {
      "id": 12345,
      "title": "Pasta alla Marinara",
      "ingredients": "[\"tomato\", \"garlic\", \"olive oil\", \"basil\"]",
      "instructions": "...",
      "similarity_score": 0.82,
      "match_percent": 91.17
    }
  ]
}
```

---

## Como o Sistema de Recomendação Funciona

1. O texto de entrada é convertido em um embedding de 384 dimensões usando MiniLM-L6-v2.
2. O embedding é normalizado em L2, permitindo que FAISS interprete o produto interno como similaridade de cosseno.
3. O índice FAISS retorna os vetores mais próximos.
4. Cada índice é mapeado para suas informações reais no arquivo `metadata.csv`.
5. A API devolve uma lista de receitas ordenada por relevância.

---

## CORS

O backend já vem configurado para aceitar requisições do frontend (React/Vite), com origens permitidas definidas no `.env`.

---

## Dicas de Desenvolvimento

* A documentação automática fica disponível em:
  `http://localhost:8000/docs`
* Para testar rapidamente sem frontend, use `curl` ou ferramentas como Postman e Thunder Client.
* Se o recomendador falhar no carregamento, a API continuará funcionando, mas com status `"degraded"`.

---

## Possíveis Problemas e Soluções

**Recommender not initialized**
Ocorre quando FAISS ou o CSV não podem ser carregados. Verifique os caminhos no `.env`.

**Erro ao gerar embedding**
Certifique-se de que o modelo MiniLM-L6-v2 está disponível. Ele será baixado automaticamente na primeira execução.

**Frontend não consegue acessar o backend**
Confirme se a origem está na variável `FRONTEND_ORIGINS`.

---

## Licença

Este projeto está sob a licença MIT.

---

## Autoria

Backend desenvolvido por **Juliana C. M. Santos**, como parte de um projeto pessoal para estudos em Machine Learning, NLP, arquitetura de APIs e recomendação baseada em embeddings.
