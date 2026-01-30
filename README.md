# Job Engine

AI-powered opportunity finder. Minimum effort, maximum return.

## What It Does

Searches everything - VC opportunities, grants, job boards, freelance gigs - and ranks them by the only metric that matters: **dollars per hour of your life**.

### Core Philosophy
- Less work, more money
- Remote by default
- AI does the hunting, you do the winning

## Stack

- **LangGraph** - Intelligent agent orchestration
- **ChromaDB** - Vector memory (remembers what you've seen, learns your preferences)
- **OpenAI** - Embeddings + reasoning
- **Brave Search** - Real-time web search
- **Perplexity** - Deep research on opportunities
- **FastAPI** - Lean API layer
- **Fly.io** - Deploys anywhere

## Quick Start

```bash
# Install
pip install -e .

# Set your API keys
cp .env.example .env
# Edit .env with your keys

# Search (CLI)
python -m src.cli search "senior python engineer remote"

# Or start the API
python -m src.cli serve
```

## Usage

### CLI

```bash
# Quick search
python -m src.cli search "AI engineer" --quick

# Full search with deep analysis
python -m src.cli search "machine learning contract work"

# Create your profile
python -m src.cli profile --income 150000 --hours 25 --skills "python,ml,ai"

# Use your profile
python -m src.cli search "remote work" --profile profile.json
```

### API

```bash
# Start server
python -m src.cli serve

# Set your profile
curl -X POST http://localhost:8000/profile \
  -H "Content-Type: application/json" \
  -d '{
    "min_income": 150000,
    "max_hours_weekly": 20,
    "skills": ["python", "ai", "ml"],
    "remote_only": true
  }'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "senior AI engineer remote"}'
```

## Configuration

Create `.env`:

```
OPENAI_API_KEY=sk-...
BRAVE_API_KEY=BSA...
PERPLEXITY_API_KEY=pplx-...
```

## Deploy

```bash
# Fly.io
fly launch
fly secrets set OPENAI_API_KEY=sk-... BRAVE_API_KEY=... PERPLEXITY_API_KEY=...
fly deploy
```

## How It Works

1. **Profile** - You tell it what you want (income, hours, skills)
2. **Search** - LangGraph agent searches Brave, Perplexity, job boards
3. **Rank** - Scores by income potential / effort required
4. **Research** - Deep dives top candidates with Perplexity
5. **Recommend** - Returns ranked opportunities with efficiency metrics

## Project Structure

```
src/
├── agents/          # LangGraph orchestrator
├── memory/          # ChromaDB vector storage
├── search/          # Brave, Perplexity integrations
├── ranking/         # Scoring algorithm
├── api/             # FastAPI routes
├── cli.py           # CLI interface
└── models.py        # Data models
```

## License

MIT
