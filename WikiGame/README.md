# WikiGame Bot

An AI-powered Wikipedia game bot using GraphRAG, GLiNER2, Llama, and semantic embeddings to navigate from any Wikipedia page to any target page in the fewest hops.

## Architecture

```
wikigame/
├── core/
│   ├── wiki_fetcher.py      # Wikipedia API wrapper + HTML parser
│   └── embedder.py          # Sentence embedding + cosine similarity
├── graph/
│   ├── graph_rag.py         # GraphRAG: node/edge store + community summaries
│   └── wild_graph.py        # WildGraph: BFS/beam search traversal engine
├── search/
│   ├── gliner_filter.py     # GLiNER2: entity extraction for link pruning
│   └── ranker.py            # Scoring pipeline: embed → GLiNER → GraphRAG → Llama
├── llm/
│   └── llama_agent.py       # Llama via Ollama: hop reasoning + chain-of-thought
├── ui/
│   └── dashboard.html       # Live browser UI showing the traversal in real time
├── bot.py                   # Main entry point
├── config.py                # All tuneable parameters
└── requirements.txt
```

## Pipeline

```
For each hop:
  1. Fetch current page → extract all outbound Wikipedia links
  2. GLiNER2 → extract entities from target page, filter link candidates
  3. sentence-transformers → embed filtered links vs target
  4. GraphRAG → re-rank using neighborhood context
  5. Llama (Ollama) → final decision with chain-of-thought reasoning
  6. WildGraph → record hop, update frontier, check for cycles
  7. Repeat until target reached or max hops exceeded
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama (for Llama)

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# Pull a model (llama3 is recommended, mistral also works well)
ollama pull llama3
# or for a smaller/faster model:
ollama pull mistral
```

### 3. Run the bot

```bash
# Basic usage
python bot.py --start "Python (programming language)" --target "Alan Turing"

# With beam search (more robust, slower)
python bot.py --start "Pizza" --target "Napoleon" --strategy beam --beam-width 3

# Disable Llama (faster, pure embedding mode)
python bot.py --start "Jazz" --target "World War II" --no-llm

# With live UI
python bot.py --start "Octopus" --target "Internet" --ui
```

### 4. Optional: Tune parameters

Edit `config.py` to adjust:
- `MAX_HOPS` — give up after N hops (default: 30)
- `BEAM_WIDTH` — paths to explore in parallel (default: 3)
- `TOP_K_CANDIDATES` — links passed to Llama after filtering (default: 10)
- `GLINER_THRESHOLD` — entity match confidence cutoff (default: 0.3)
- `OLLAMA_MODEL` — which Ollama model to use (default: "llama3")

## How It Works

### GLiNER2 Filtering
GLiNER2 extracts named entities (people, places, concepts, orgs) from the **target** page. When scoring link candidates on the current page, any link whose anchor text or title matches a target entity gets a significant score boost. This prunes ~200 links down to ~20 meaningful candidates before expensive embedding.

### GraphRAG
As the bot visits pages, it builds a local knowledge graph. Nodes = pages (with embeddings of their lede paragraph). Edges = links. GraphRAG computes "community" clusters of semantically related pages. When choosing a hop, it can detect if a candidate link is in the same community as the target — a strong signal you're heading the right direction.

### Llama Chain-of-Thought
The top-K candidates from embedding + GraphRAG are passed to Llama with a structured prompt:
```
You are navigating Wikipedia to reach "[TARGET]".
You are currently on "[CURRENT PAGE]".
Here are your top link candidates with scores:
  1. [Link A] (score: 0.82) — "A is a concept in physics..."
  2. [Link B] (score: 0.79) — "B was a mathematician who..."
  ...
Think step by step about which link brings you closer to "[TARGET]".
Return ONLY the number of your choice.
```

### WildGraph Traversal
Implements beam search with cycle detection. Maintains a priority queue of paths ranked by cumulative score. Visited nodes are penalized (not hard-blocked, since sometimes revisiting a hub page is strategically useful).

## Example Output

```
🎯 Target: Alan Turing
📖 Start:  Pizza

Hop 1: Pizza → Italy (score: 0.61)
  Reasoning: Italy is a broad geographic hub likely connected to mathematics and science
Hop 2: Italy → Mathematics (score: 0.74)
  Reasoning: Mathematics is the direct domain of Alan Turing's work
Hop 3: Mathematics → Computer science (score: 0.88)
  Reasoning: Computer science is Turing's field
Hop 4: Computer science → Alan Turing (score: 0.99) ✅

✅ Reached target in 4 hops!
Path: Pizza → Italy → Mathematics → Computer science → Alan Turing
```
