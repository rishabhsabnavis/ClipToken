# ContextOS

A drop-in middleware layer that compresses, deduplicates, and restructures LLM agent context before each API call. Agents route messages through ContextOS instead of directly to the LLM. Compression is transparent -- the agent sees normal responses.

**What makes ContextOS different:** most context compressors apply fixed, blind rules (compress anything older than N turns, hit a target ratio). ContextOS instead compresses based on **measured downstream task impact**, and it **gets less lossy over time** -- it keeps the original content recoverable, detects when a past compression actually hurt the agent, and learns to stop making that mistake. The compression policy improves with every session.

---

## Project goal

Reduce token usage for long-running LLM agents by 60-90% with no degradation in task completion quality -- and drive information loss **toward zero over time** as the system learns which compressions are safe. Expose a FastAPI server that mirrors the Anthropic messages API surface so any agent can swap in ContextOS with a one-line URL change.

The headline result this project must produce: a single plot showing **loss rate falling across repeated sessions while compression ratio stays high.**

---

## Architecture

ContextOS has two paths: a **request path** that compresses each call (Modules 1-5), and a **learning loop** that runs across calls and makes the request path less lossy over time (FidelityStore, LossDetector, CompressionPolicy).

```
Agent (LangGraph / raw SDK / any framework)
    |
    v
ContextOS FastAPI server  (POST /v1/messages)
    |
    |== REQUEST PATH (per call) ===========================
    |   Module 1: Tool result compressor
    |   Module 2: Semantic deduplicator
    |   Module 3: Symbol substitution
    |   Module 4: Adaptive compressor  <-- consults CompressionPolicy
    |   Module 5: Context assembler
    |
    |== LEARNING LOOP (across calls) ======================
    |   FidelityStore   -- keeps originals; makes loss reversible (Layer A)
    |   LossDetector    -- flags when a compression hurt the agent (Layer B)
    |   CompressionPolicy -- learns from loss events; gets less lossy (Layer C)
    |
    v
LLM provider (Anthropic / OpenAI)
    |
    v
Response passed back to agent unchanged
```

How the two paths connect: Module 4 asks the **CompressionPolicy** how aggressively to compress each segment. Every original chunk is written to the **FidelityStore** before it is compressed, so nothing is ever truly lost. The **LossDetector** watches subsequent turns for evidence that a past compression discarded something the agent needed; each detected loss event updates the policy so the same mistake is not repeated.

---

## Module contracts

Each module is a standalone Python class with a single primary method. Request-path modules are stateless except the adaptive compressor (which maintains turn history per session). The learning-loop components are stateful and persist across sessions.

### Module 1 -- ToolResultCompressor

```python
class ToolResultCompressor:
    def compress(self, tool_name: str, raw_result: dict) -> str:
        ...
```

- Checks `schema_registry` dict for known tool schemas
- If schema found: extracts only listed fields
- If no schema: calls Haiku with a summarization prompt
- Returns compressed string, always under 100 tokens
- Logs: `tool_name`, `tokens_before`, `tokens_after`
- The `schema_registry` may be **auto-populated by the learning loop**: once the LossDetector has observed which fields of a tool's output the agent actually depends on, those fields become the schema, making that tool near-lossless from then on (see CompressionPolicy)

### Module 2 -- SemanticDeduplicator

```python
class SemanticDeduplicator:
    def deduplicate(self, facts: list[str], threshold: float = 0.92) -> list[str]:
        ...
```

- Embeds all facts with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Computes pairwise cosine similarity
- Clusters pairs above threshold
- Keeps longest string per cluster
- Returns deduplicated list

### Module 3 -- SymbolSubstitutor

```python
class SymbolSubstitutor:
    def build_table(self, messages: list[dict], min_length: int = 20, min_occurrences: int = 2) -> dict:
        ...

    def substitute(self, text: str, table: dict) -> str:
        ...
```

- Scans all message content for recurring long strings
- Assigns symbols `$S1`, `$S2`, etc.
- Substitution is pure string replace -- no regex needed for MVP

### Module 4 -- AdaptiveCompressor

The differentiator. Replaces the old fixed-tier "summarization pyramid." Instead of compressing by age, it compresses each segment by its **predicted impact on the agent's next action**, choosing the most aggressive compression level whose predicted impact stays under a budget.

```python
class AdaptiveCompressor:
    def compress_history(self, turns: list[Turn], policy: CompressionPolicy) -> list[Turn]:
        ...
```

- For each `Turn`, build a feature vector (see `Segment`) and ask `policy.decide(...)` for a compression level
- Apply the chosen level: `verbatim` (no change), `bullet` (Haiku bullet summary), `sentence` (single-sentence Haiku summary), or `drop` (replaced by a FidelityStore reference marker)
- Before compressing, write the original to the FidelityStore so the decision is reversible
- During **cold start** (before the policy has data) fall back to the age-based prior: turns 0..`tier1` -> sentence, `tier1`..`tier2` -> bullet, `tier2`+ -> verbatim
- Logs: per-segment chosen level, predicted impact, `tokens_before`, `tokens_after`

```python
@dataclass
class Turn:
    index: int
    role: str        # "user" | "assistant"
    content: str
    summary: str | None = None
    level: str = "verbatim"   # "verbatim" | "bullet" | "sentence" | "drop"
    fidelity_ref: str | None = None   # content hash in FidelityStore, if compressed

@dataclass
class Segment:
    """Features the policy uses to decide a compression level."""
    turn_index: int
    age_turns: int
    token_len: int
    tool_name: str | None
    semantic_relevance: float   # cosine sim to current goal/last user turn
    times_referenced_recently: int
    segment_type: str           # "tool_result" | "assistant_reasoning" | "user" | ...
```

### Module 5 -- ContextAssembler

```python
class ContextAssembler:
    def assemble(self, turns, symbol_table, tool_results) -> list[dict]:
        ...
```

- Prepends symbol table as a system message addendum
- Inserts compressed tool results in place of raw results
- Carries FidelityStore reference markers for any dropped segments so they can be restored on demand
- Returns final messages list ready to send to LLM

---

## Learning loop (the "less loss over time" system)

These three components are what let ContextOS improve. They are not on the hot path of every call -- they run alongside it and persist state across sessions.

### Layer A -- FidelityStore (loss is reversible)

```python
class FidelityStore:
    def put(self, content: str) -> str:        # returns content hash
        ...
    def get(self, content_hash: str) -> str:   # restores original
        ...
```

- Every original chunk is stored before compression, keyed by `sha256(content)`
- The wire payload carries only the compressed form (plus a `⟨ref:hash⟩` marker for dropped segments)
- Makes "loss" really "deferred detail" -- the full content is always recoverable
- Backend: SQLite on disk for MVP, Redis later
- This is what lets the README say *compressed in transit, full fidelity recoverable* instead of *lossy*

### Layer B -- LossDetector (find the mistakes)

```python
class LossDetector:
    def scan(self, session: Session, response: dict) -> list[LossEvent]:
        ...
```

Detects evidence that a past compression discarded something the agent needed. Three signals:

- **Re-request:** the agent re-runs a tool / re-reads a file whose result was already compressed -> attributes a loss event to that segment (free, strong signal)
- **Shadow divergence:** on a sampled fraction of turns (`CONTEXTOS_SHADOW_SAMPLE_RATE`), also run the raw uncompressed context and compare the next action (tool name + args) or response embedding; divergence above a threshold -> loss event
- **Judge flag:** optional cheap Haiku call asking whether the compressed context lost anything that changed the answer

Each `LossEvent` records the offending `segment_type`, `tool_name`, and the compression `level` that caused it.

### Layer C -- CompressionPolicy (learn, get less lossy)

```python
class CompressionPolicy:
    def decide(self, segment: Segment) -> str:            # returns a compression level
        ...
    def update(self, events: list[LossEvent]) -> None:    # learn from outcomes
        ...
    def save(self) -> None: ...
    def load(self) -> None: ...
```

- `decide` picks the most aggressive level whose predicted impact < `CONTEXTOS_IMPACT_BUDGET`
- `update` raises the protection of segment types / tools that have caused loss events, and promotes reliably-safe fields of a tool into Module 1's `schema_registry`
- MVP form: online-updated per-`(segment_type, tool_name)` loss-propensity estimates feeding a thresholded decision. Upgrade path: a small sklearn logistic-regression / contextual bandit once enough data exists
- Persists to `CONTEXTOS_POLICY_PATH` so learning survives restarts and accumulates across sessions

---

## API surface

### POST /v1/messages

Mirrors the Anthropic messages API exactly.

Request body:
```json
{
  "model": "claude-sonnet-4-6",
  "max_tokens": 1000,
  "messages": [...],
  "tools": [...],
  "system": "..."
}
```

Response: pass-through from Anthropic, unchanged.

Headers required:
- `x-api-key`: Anthropic API key (forwarded)
- `contextos-session-id`: used to key session state (turn history, learning loop) per session

### GET /v1/stats/{session_id}

Returns compression and learning stats for a session:
```json
{
  "session_id": "...",
  "turns": 12,
  "tokens_before": 8400,
  "tokens_after": 610,
  "compression_ratio": 0.93,
  "modules_fired": ["compressor", "dedup", "symbols", "adaptive"],
  "loss_events": 1,
  "loss_rate": 0.08,
  "sessions_seen": 37
}
```

### GET /v1/loss-curve

Returns the headline artifact: `loss_rate` and `compression_ratio` aggregated per session index across the lifetime of the policy, for plotting "loss falls over time while compression stays high."

---

## Tech stack

| Concern | Library |
|---|---|
| API server | FastAPI + uvicorn |
| Request/response models | Pydantic v2 |
| Token counting | tiktoken |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Similarity | scikit-learn cosine_similarity |
| Learned policy | scikit-learn (logistic regression / bandit) |
| LLM calls | anthropic SDK |
| FidelityStore | SQLite (in-memory dict acceptable for first MVP, Redis later) |
| Session + policy state | Python dict in-memory; policy persisted to disk |
| Testing | pytest |

Python version: 3.11+

---

## File structure

```
contextos/
  CLAUDE.md
  README.md
  pyproject.toml
  main.py                  # FastAPI app, single entry point
  pipeline.py              # Orchestrates the request path (modules 1-5) + learning loop
  modules/
    compressor.py          # Module 1
    deduplicator.py        # Module 2
    substitutor.py         # Module 3
    adaptive.py            # Module 4 (AdaptiveCompressor)
    assembler.py           # Module 5
  learning/
    store.py               # Layer A -- FidelityStore
    detector.py            # Layer B -- LossDetector
    policy.py              # Layer C -- CompressionPolicy
  schemas/
    registry.py            # Tool name -> field list mappings (seedable + auto-learned)
    models.py              # Pydantic request/response models, Turn, Segment, LossEvent
  eval/
    test_pipeline.py       # End-to-end eval with token count assertions
    fixtures/
      sample_conversation.json   # 10-turn research agent conversation
  benchmarks/
    loss_over_time.py      # Replays sessions, emits the loss-vs-sessions curve
```

---

## Environment variables

```
ANTHROPIC_API_KEY=sk-ant-...
CONTEXTOS_PORT=8000
CONTEXTOS_DEDUP_THRESHOLD=0.92
CONTEXTOS_SYMBOL_MIN_LENGTH=20
CONTEXTOS_SYMBOL_MIN_OCCURRENCES=2
CONTEXTOS_IMPACT_BUDGET=0.15          # max predicted task-impact the policy will accept
CONTEXTOS_SHADOW_SAMPLE_RATE=0.05     # fraction of turns to run raw for loss detection
CONTEXTOS_FIDELITY_STORE_PATH=./.contextos/fidelity.db
CONTEXTOS_POLICY_PATH=./.contextos/policy.json
CONTEXTOS_COLDSTART_TIER1=5           # age-based prior used before the policy has data
CONTEXTOS_COLDSTART_TIER2=10
CONTEXTOS_COMPRESS_MODEL=claude-haiku-4-5-20251001
```

---

## Build order

Build and test each piece independently before wiring together:

1. `modules/substitutor.py` -- pure Python, no dependencies, easiest start
2. `modules/compressor.py` -- needs Anthropic SDK and tiktoken
3. `modules/deduplicator.py` -- needs sentence-transformers and sklearn
4. `learning/store.py` -- FidelityStore (SQLite); simple and unblocks reversibility
5. `modules/adaptive.py` -- AdaptiveCompressor with the cold-start age prior first (no learning yet)
6. `modules/assembler.py` -- pure Python, combines outputs of 1-4
7. `pipeline.py` -- wires the request path end to end
8. `main.py` -- wraps pipeline in FastAPI
9. `learning/detector.py` -- LossDetector (re-request signal first, then shadow run)
10. `learning/policy.py` -- CompressionPolicy; close the loop so loss drops over time
11. `eval/test_pipeline.py` -- end-to-end with assertions
12. `benchmarks/loss_over_time.py` -- produce the headline loss-vs-sessions curve

Note the staging: the request path (1-8) is a working compressor on its own. The learning loop (9-12) is what makes it improve over time -- build it once the path works.

---

## Eval criteria

A compression is considered passing if:

- Token reduction >= 50% vs raw
- All key facts from the original conversation are present in the compressed output (checked by an LLM judge call)
- LLM response on the compressed context is semantically equivalent to response on raw context (cosine similarity >= 0.85 between response embeddings)

The learning loop is considered working if:

- **Loss rate decreases across repeated sessions** on the benchmark (the headline result), while compression ratio stays >= 50%
- Any segment dropped from the wire is byte-for-byte recoverable from the FidelityStore (reversibility test)

Run evals with:
```bash
pytest eval/test_pipeline.py -v
python benchmarks/loss_over_time.py   # emits the loss-vs-sessions curve
```

---

## Coding conventions

- Type hints on every function signature
- Docstring on every class and public method
- No global state -- everything passed explicitly
- Each module logs token counts before and after (use Python `logging`, not print)
- Raise specific exceptions, not generic `Exception`
- Keep Haiku calls under 200 input tokens -- compress the prompt too

---

## Known tradeoffs

- Compressed **in transit but recoverable** -- the FidelityStore keeps originals, so on-the-wire loss is reversible. True permanent loss only happens if the store is dropped. Tasks needing exact wording every turn (legal, medical) should run with a high impact budget / verbatim bias
- **Cold start** -- early sessions are lossier; the policy needs observed loss events before it converges. The loss-vs-sessions curve is expected to start higher and fall
- **Shadow runs cost tokens** -- loss detection via raw-context comparison is sampled (`CONTEXTOS_SHADOW_SAMPLE_RATE`), not run every turn, to keep overhead small
- Embedding model is English-optimized -- multilingual content degrades dedup quality
- In-memory session state resets on server restart -- the learned policy is persisted to disk, but per-session turn history is not (add Redis for production)
- Symbol table costs tokens upfront -- only profitable after 2-3 repetitions of each symbol
