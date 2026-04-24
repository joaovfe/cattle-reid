# Cattle Re-ID Drone

Sistema de identificação individual de bovinos a partir de imagens de drone.

**Pipeline**: Detection (YOLO) → Tracking (ByteTrack) → Oriented Crop → Embedding (DINOv2/DINOv3) → FAISS → Tracklet Aggregation → Identity Decision.

## Estrutura do projeto

- **`src/`** — Código da aplicação e API
  - `src/ai/` — Pipeline ML: detection, tracking, oriented_crop, embedding, reid, events
  - `src/api/` — FastAPI (enrollment, inferência, eventos, health)
  - `src/training/` — Scripts de treino YOLO e Re-ID
  - `src/scripts/` — CLI (inferência em vídeo)
- **`core/`** — Banco de dados e dados
  - `core/database/` — Modelos SQLAlchemy, sessão async, migrations Alembic
  - `core/data/` — Loaders de dataset (YOLO, BECA, etc.), índice FAISS (`core/data/faiss_index`)
- **`configs/`** — YAML de configuração
- **`docker-compose.yml`** — Apenas PostgreSQL (sem Dockerfiles para a app)

## Pré-requisitos

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) para dependências: `pip install uv`
- (Opcional) GPU NVIDIA com CUDA

## Instalação

```bash
cd cattle-reid-drone
uv sync
```

## Como rodar

### 1. Subir o banco (PostgreSQL)

```bash
docker compose up -d
```

Crie as tabelas (primeira vez):

```bash
export DATABASE_URL="postgresql+asyncpg://cattle:cattle@localhost:5432/cattle_reid"
uv run alembic -c alembic.ini upgrade head
```

(O Alembic usa `core/database/migrations`; a URL sync para migrations é obtida de `DATABASE_URL` convertendo `postgresql+asyncpg` → `postgresql`.)

### 2. API FastAPI

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

- Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  
- Endpoints: `POST /animals`, `POST /animals/{id}/enroll`, `GET /animals/{id}`, `POST /inference/video`, `POST /inference/frames`, `GET /events`

### 3. Inferência em vídeo (CLI)

```bash
uv run python -m src.scripts.run_inference_video --video /caminho/para/video.mp4
```

Com índice FAISS e saída JSON:

```bash
uv run python -m src.scripts.run_inference_video --video video.mp4 --faiss core/data/faiss_index --output resultado.json
```

### 3.1 CLI simplificado (estilo Agro Vision)

Use o comando abaixo para abrir o fluxo de **analisar video**:

```bash
uv run cattle-reid-drone
```

Esse comando usa a pasta `videos/` na raiz do projeto. Basta colocar os videos ali e escolher um da lista para analisar.

Antes de analisar, o CLI atualiza automaticamente a galeria de embeddings usando `gallery_images/`:

Antes de analisar, o CLI sincroniza o FAISS a partir do **banco**:

- Fonte oficial: tabela `animal_crops` (imagens vinculadas ao `animal_id`)
- Cada embedding usa o ID já salvo no banco
- O pipeline reconstrói a galeria FAISS com base nesses registros

### 4. Treino YOLO (detecção)

```bash
uv run python -m src.training.train_yolo --data path/to/data.yaml --epochs 50
```

### 5. Treino Re-ID (encoder)

```bash
uv run python -m src.training.train_reid --data_root path/to/BECA-D --epochs 30
```

## Embedding: DINOv2 vs DINOv3

- **DINOv3** é o **padrão** (`facebook/dinov3-vits16-pretrain-lvd1689m` no `configs/default.yaml`). O cartão do modelo no Hugging Face pode ser *gated* (login + aceite de licença) e exige `HF_TOKEN`.
- **DINOv2** (`facebook/dinov2-small`) continua disponível: altere `models.embedding.model_name` no YAML ou use `CattleEmbeddingEncoder(model_name="facebook/dinov2-small")`.

A API e a CLI leem `models.embedding` do config ao instanciar `CattleEmbeddingEncoder`. Ao mudar entre v2 e v3, **reconstrua o índice FAISS** (embeddings de dimensão/espaço diferentes).

## Variáveis de ambiente

- `DATABASE_URL` — ex.: `postgresql+asyncpg://cattle:cattle@localhost:5432/cattle_reid`
- `FAISS_INDEX_PATH` — ex.: `./core/data/faiss_index` (opcional; padrão usa `core/data/faiss_index`)

## Resumo rápido

```bash
uv sync
docker compose up -d
export DATABASE_URL="postgresql+asyncpg://cattle:cattle@localhost:5432/cattle_reid"
uv run alembic -c alembic.ini upgrade head
uv run uvicorn src.api.main:app --reload --port 8000
# Noutro terminal:
uv run python -m src.scripts.run_inference_video --video seu_video.mp4
```
