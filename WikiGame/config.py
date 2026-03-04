# config.py — All tuneable parameters for the WikiGame Bot

# ─── Wikipedia API ──────────────────────────────────────────────────────────
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_REQUEST_DELAY = 0.5        # seconds between API calls (be polite)
WIKI_USER_AGENT = "WikiGameBot/1.0 (https://github.com/you/wikigame)"
WIKI_LEDE_SENTENCES = 5         # how many sentences of intro paragraph to embed

# ─── Traversal ───────────────────────────────────────────────────────────────
MAX_HOPS = 30                   # give up after this many hops
BEAM_WIDTH = 3                  # number of parallel paths in beam search
STRATEGY = "beam"               # "greedy" | "beam"
CYCLE_PENALTY = 0.5             # score multiplier for already-visited pages

# ─── Embedding ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim, great for semantic search
# Alternatives:
#   "all-mpnet-base-v2"         — higher quality, slower (768-dim)
#   "multi-qa-MiniLM-L6-cos-v1" — tuned for retrieval tasks

# ─── GLiNER2 ─────────────────────────────────────────────────────────────────
GLINER_MODEL = "urchade/gliner_medium-v2.1"
GLINER_THRESHOLD = 0.3          # min entity confidence to keep
GLINER_ENTITY_TYPES = [         # entity types to extract from target page
    "person", "location", "organization",
    "concept", "event", "scientific field",
    "technology", "country", "era"
]
GLINER_SCORE_BOOST = 0.25       # score bonus for entity-matched link candidates
MAX_LINKS_AFTER_GLINER = 25     # max candidates passed to embedding stage

# ─── GraphRAG ────────────────────────────────────────────────────────────────
GRAPHRAG_COMMUNITY_WEIGHT = 0.15    # how much community membership affects score
GRAPHRAG_NEIGHBOR_WEIGHT = 0.10     # bonus for links that are graph-neighbors of target
GRAPHRAG_MIN_COMMUNITY_SIZE = 3     # ignore tiny communities

# ─── Llama / Ollama ──────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3"         # "mistral", "llama3", "llama3.1", "phi3", etc.
OLLAMA_TIMEOUT = 30             # seconds
TOP_K_FOR_LLM = 10              # how many candidates to pass to Llama
LLM_TEMPERATURE = 0.1           # low = deterministic; high = creative
USE_LLM = True                  # set False for pure embedding mode (faster)

# ─── Scoring weights ─────────────────────────────────────────────────────────
# Final score = weighted sum of all signals
WEIGHT_EMBEDDING = 0.50
WEIGHT_GLINER = 0.20
WEIGHT_GRAPHRAG = 0.15
WEIGHT_LLM = 0.15               # only applies to top-K after LLM reranking

# ─── UI ──────────────────────────────────────────────────────────────────────
UI_PORT = 8765                  # WebSocket port for live dashboard
SHOW_UI = False                 # launch browser dashboard

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"              # "DEBUG" | "INFO" | "WARNING"
LOG_FILE = "wikigame.log"
VERBOSE_SCORES = False          # print per-candidate scores each hop
