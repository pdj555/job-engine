# Job Engine

Find opportunities. Ranked by **$/hour**.

## Use

```bash
pip install -e .

# Set API keys
export OPENAI_API_KEY=sk-...
export BRAVE_API_KEY=BSA...      # optional
export PERPLEXITY_API_KEY=pplx-... # optional

# Search
job-engine "AI engineer"
job-engine "python freelance"
job-engine "startup equity"
```

## Output

```
#   Title                                    Company        Pay       Hrs    $/hr
1   Senior ML Engineer (Remote)              Acme AI        $220,000  30     $147
2   AI Consultant - Part Time                TechCorp       $180,000  20     $180
3   Python Contract - 6 months               StartupX       $150,000  25     $120
...

Top picks:
  1. Senior ML Engineer (Remote)
     https://example.com/job/123
     $147/hr
```

## API

```bash
# Start server
job-engine serve

# Search
curl "localhost:8000/search?q=AI+engineer"
```

## How It Works

1. Searches Brave + Perplexity in parallel
2. Extracts pay/hours with GPT-4o
3. Ranks by efficiency: `pay / (hours * 50 weeks)`
4. Penalizes office jobs (-30%)
5. Returns highest $/hr first

## Files

```
src/
├── engine.py   # The whole thing
├── models.py   # One model: Opportunity
├── cli.py      # CLI
└── api/        # FastAPI
```

## Deploy

```bash
fly launch
fly secrets set OPENAI_API_KEY=... BRAVE_API_KEY=... PERPLEXITY_API_KEY=...
fly deploy
```
