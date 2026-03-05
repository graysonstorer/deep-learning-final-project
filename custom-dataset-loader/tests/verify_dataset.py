from __future__ import annotations

import gc
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "page_embeddings.pt"
PAGES_PATH = PROJECT_ROOT / "data" / "processed" / "pages_sanitized.jsonl"
LINKS_PATH = PROJECT_ROOT / "data" / "processed" / "links_table.jsonl"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset_metadata.json"

# Graph artifact naming varies by repo; use the canonical one produced by graph/build_page_graph.py.
GRAPH_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "page_graph.gpickle",
    PROJECT_ROOT / "data" / "processed" / "page_graph.pkl",
    PROJECT_ROOT / "data" / "graph" / "page_graph.pt",
    PROJECT_ROOT / "data" / "graph" / "page_graph.gpickle",
]


def _divider(title: str) -> None:
    print("\n" + "=" * 30)
    print(f"DATASET VERIFICATION: {title}")
    print("=" * 30)


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def _file_size_mb(path: Path) -> str:
    return _fmt_mb(path.stat().st_size)


def _maxrss_mb() -> Optional[float]:
    """
    Best-effort process memory.
    - On macOS, ru_maxrss is bytes.
    - On Linux, ru_maxrss is kilobytes.
    """
    try:
        import resource  # unix only

        ru = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = float(getattr(ru, "ru_maxrss", 0.0))
        if maxrss <= 0:
            return None
        if sys.platform == "darwin":
            return maxrss / (1024 * 1024)
        return (maxrss * 1024) / (1024 * 1024)
    except Exception:
        return None


def _print_mem(prefix: str = "") -> None:
    rss = _maxrss_mb()
    if rss is None:
        return
    print(f"{prefix}maxrss≈{rss:.1f} MB")


def _assert(path: bool, msg: str) -> None:
    if not path:
        raise AssertionError(msg)


def _load_graph(path: Path) -> nx.DiGraph:
    read_gpickle = getattr(nx, "read_gpickle", None)
    if callable(read_gpickle) and path.suffix in {".gpickle", ".pkl"}:
        G = read_gpickle(path)
        if not isinstance(G, nx.DiGraph):
            raise TypeError(f"Unexpected object type in {path}: {type(G)}")
        return G

    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, nx.DiGraph):
        raise TypeError(f"Unexpected object type in {path}: {type(obj)}")
    return obj


def _choose_graph_path() -> Path:
    for p in GRAPH_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No graph artifact found. Looked for:\n"
        + "\n".join(f"- {p}" for p in GRAPH_CANDIDATES)
        + "\nRun: python3 graph/build_page_graph.py"
    )


def _describe_tensor(name: str, t: Any) -> None:
    if torch.is_tensor(t):
        shape = list(t.shape)
        dev = str(t.device)
        print(f"{name:18s}: {t.__class__.__name__} dtype={t.dtype} device={dev} shape={shape}")
        return
    print(f"{name:18s}: {type(t)} (non-tensor)")


def _resolve_training_dataset_class() -> Optional[Type[Dataset]]:
    """
    Best-effort import hook for a future canonical training dataset.
    If you later add a dataset class, put it under one of these imports.
    """
    candidates: List[Tuple[str, str]] = [
        ("training.dataset", "WikipediaGraphDataset"),
        ("training.wikipedia_dataset", "WikipediaGraphDataset"),
        ("training.data", "WikipediaGraphDataset"),
        ("training.dataset", "Dataset"),
    ]
    for mod_name, cls_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if isinstance(cls, type) and issubclass(cls, Dataset):
                return cls
        except Exception:
            continue
    return None


class GraphEmbeddingDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    A batch-safe Dataset view over the canonical artifacts:
    - data/embeddings/page_embeddings.pt
    - data/processed/page_graph.gpickle

    Each sample is a single page with a fixed-length embedding and a fixed-length
    neighbor list (padded) so default PyTorch DataLoader collation works.
    """

    def __init__(
        self,
        embeddings_obj: Dict[str, Any],
        G: nx.DiGraph,
        k_neighbors: int = 16,
        seed: int = 123,
    ) -> None:
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        _assert(self.k_neighbors > 0, "k_neighbors must be > 0")

        page_ids_raw = embeddings_obj.get("page_ids")
        embeddings = embeddings_obj.get("embeddings")
        _assert(isinstance(page_ids_raw, list), "embeddings_obj['page_ids'] must be a list")
        _assert(torch.is_tensor(embeddings), "embeddings_obj['embeddings'] must be a torch.Tensor")
        _assert(embeddings.ndim == 2, "embeddings tensor must be 2D (N, D)")
        _assert(len(page_ids_raw) == int(embeddings.shape[0]), "page_ids length must match embeddings rows")

        # Normalize page_ids and build index mapping.
        page_ids: List[int] = []
        for x in page_ids_raw:
            try:
                page_ids.append(int(x))
            except Exception as e:
                raise TypeError(f"Non-int page_id in embeddings_obj['page_ids']: {x!r}") from e

        if len(set(page_ids)) != len(page_ids):
            raise ValueError("Duplicate page_ids found in embeddings artifact")

        self._embeddings = embeddings.detach().cpu().float()
        self._emb_index: Dict[int, int] = {pid: i for i, pid in enumerate(page_ids)}

        # Keep only nodes present in BOTH embeddings and graph.
        graph_node_ids = set()
        for n in G.nodes():
            try:
                graph_node_ids.add(int(n))
            except Exception:
                continue

        self.page_ids: List[int] = sorted(pid for pid in page_ids if pid in graph_node_ids)
        _assert(len(self.page_ids) > 0, "No overlapping page IDs between embeddings and graph nodes")

        # Precompute deterministic neighbor lists (top-weighted outgoing neighbors).
        adjacency: Dict[int, List[Tuple[int, float]]] = {}
        for u, v, d in G.edges(data=True):
            try:
                uu = int(u)
                vv = int(v)
            except Exception:
                continue
            if uu not in graph_node_ids or vv not in graph_node_ids:
                continue
            w = d.get("weight", 1.0)
            try:
                ww = float(w)
            except Exception:
                continue
            adjacency.setdefault(uu, []).append((vv, ww))

        for u, lst in adjacency.items():
            # Stable ordering: by weight desc, then id asc.
            lst.sort(key=lambda x: (-x[1], x[0]))

        self._adjacency = adjacency
        self._rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.page_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = int(self.page_ids[int(idx)])
        emb_row = int(self._emb_index[pid])
        x = self._embeddings[emb_row]  # (D,)

        neighbors = self._adjacency.get(pid, [])
        # If a node has many neighbors, take the top ones; if it has fewer, pad.
        picked = neighbors[: self.k_neighbors]
        neigh_ids = [int(v) for v, _ in picked]
        weights = [float(w) for _, w in picked]

        pad_n = self.k_neighbors - len(neigh_ids)
        if pad_n > 0:
            neigh_ids.extend([-1] * pad_n)
            weights.extend([0.0] * pad_n)

        neighbor_page_ids = torch.tensor(neigh_ids, dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        neighbor_mask = neighbor_page_ids != -1

        # Fixed-shape edge_index for batch-safety. Use edge_mask to ignore padded edges.
        # Subgraph node indices: 0 = center, 1..k = neighbor slots.
        src = torch.zeros(self.k_neighbors, dtype=torch.long)
        dst = torch.arange(1, self.k_neighbors + 1, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)  # (2, k)
        edge_mask = neighbor_mask.clone()

        return {
            "page_id": torch.tensor(pid, dtype=torch.long),
            "node_features": x,  # (D,)
            "neighbor_page_ids": neighbor_page_ids,  # (k,)
            "edge_weight": edge_weight,  # (k,)
            "neighbor_mask": neighbor_mask,  # (k,)
            "edge_index": edge_index,  # (2, k)
            "edge_mask": edge_mask,  # (k,)
        }


@dataclass
class PhaseResult:
    name: str
    ok: bool
    detail: str = ""


def _run_phase(name: str, fn: Callable[[], None]) -> PhaseResult:
    _divider(name)
    t0 = time.time()
    try:
        fn()
    except Exception as e:
        dt = time.time() - t0
        print(f"✗ FAIL ({dt:.2f}s): {type(e).__name__}: {e}")
        return PhaseResult(name=name, ok=False, detail=f"{type(e).__name__}: {e}")
    dt = time.time() - t0
    print(f"✓ PASS ({dt:.2f}s)")
    return PhaseResult(name=name, ok=True)


def main() -> None:
    results: List[PhaseResult] = []

    # Shared objects across phases
    artifacts: Dict[str, Path] = {}
    embeddings_obj: Dict[str, Any] = {}
    G: nx.DiGraph = nx.DiGraph()
    dataset: Optional[Dataset] = None
    sample0: Optional[Dict[str, Any]] = None
    batch0: Optional[Dict[str, Any]] = None

    def phase_artifacts() -> None:
        nonlocal artifacts
        graph_path = _choose_graph_path()
        artifacts = {
            "embeddings": EMBEDDINGS_PATH,
            "graph": graph_path,
            "pages_sanitized": PAGES_PATH,
            "links_table": LINKS_PATH,
            "metadata": METADATA_PATH,
        }

        for k, p in artifacts.items():
            _assert(p.exists(), f"Missing required artifact: {k} at {p}")

        for k, p in artifacts.items():
            print(f"✓ {k} found ({_file_size_mb(p)}) — {p.relative_to(PROJECT_ROOT)}")
        _print_mem(prefix="After discovery: ")

    results.append(_run_phase("ARTIFACT DISCOVERY", phase_artifacts))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_loading() -> None:
        nonlocal embeddings_obj, G
        t0 = time.time()
        embeddings_obj = torch.load(artifacts["embeddings"], map_location="cpu")
        dt = time.time() - t0
        _assert(isinstance(embeddings_obj, dict), "Embeddings artifact must be a dict (torch.save payload)")
        _assert("page_ids" in embeddings_obj and "embeddings" in embeddings_obj, "Embeddings dict missing keys")
        page_ids = embeddings_obj["page_ids"]
        emb = embeddings_obj["embeddings"]
        _assert(isinstance(page_ids, list), "embeddings_obj['page_ids'] must be a list")
        _assert(torch.is_tensor(emb), "embeddings_obj['embeddings'] must be a tensor")
        _assert(emb.ndim == 2, "embeddings tensor must be 2D (N, D)")
        _assert(len(page_ids) == int(emb.shape[0]), "page_ids length must equal embeddings rows")
        print(f"Loaded embeddings in {dt:.2f}s")
        print(f'Embeddings tensor shape: {tuple(emb.shape)} dtype={emb.dtype}')
        est_mb = (emb.numel() * emb.element_size()) / (1024 * 1024)
        print(f"Embeddings tensor size (approx): {est_mb:.2f} MB")
        if int(emb.shape[1]) != 384:
            print(f"[warn] embedding dim is {int(emb.shape[1])}, expected 384 for all-MiniLM-L6-v2")

        t1 = time.time()
        G = _load_graph(artifacts["graph"])
        dtg = time.time() - t1
        print(f"Loaded graph in {dtg:.2f}s from {artifacts['graph'].relative_to(PROJECT_ROOT)}")
        print(f"Graph type: {type(G).__name__}")
        print(f"Graph nodes: {G.number_of_nodes():,}")
        print(f"Graph edges: {G.number_of_edges():,}")
        _print_mem(prefix="After loading: ")

    results.append(_run_phase("DATASET LOADING", phase_loading))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_structural_validation() -> None:
        _assert(isinstance(G, nx.DiGraph), f"Graph must be nx.DiGraph, got {type(G)}")
        _assert(G.number_of_nodes() > 0, "Graph has no nodes")
        _assert(G.number_of_edges() > 0, "Graph has no edges")

        self_loops = list(nx.selfloop_edges(G))
        _assert(len(self_loops) == 0, f"Graph contains self-loops (count={len(self_loops)})")
        print("✓ no self-loops")

        # Check weights
        invalid = 0
        min_w = math.inf
        max_w = -math.inf
        for _, _, d in G.edges(data=True):
            w = d.get("weight", None)
            if not isinstance(w, (int, float)) or not math.isfinite(float(w)) or float(w) <= 0.0:
                invalid += 1
                if invalid <= 5:
                    print(f"[bad] invalid edge weight attrs={d}")
                continue
            ww = float(w)
            min_w = min(min_w, ww)
            max_w = max(max_w, ww)
        _assert(invalid == 0, f"Found {invalid} edges with invalid weight")
        print(f"✓ edge weights valid (min={min_w:.4f}, max={max_w:.4f})")

        # Connectivity signal (undirected giant component)
        UG = G.to_undirected()
        comps = list(nx.connected_components(UG))
        _assert(len(comps) > 0, "Graph has no connected components (unexpected)")
        largest = max(comps, key=len)
        ratio = len(largest) / G.number_of_nodes()
        print(f"Giant component ratio (undirected): {ratio:.3f}")
        _assert(ratio > 0.5, "Graph is too fragmented (giant component ratio <= 0.5)")
        _print_mem(prefix="After structural validation: ")

    results.append(_run_phase("STRUCTURAL VALIDATION", phase_structural_validation))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_metadata_validation() -> None:
        _assert(METADATA_PATH.exists(), f"Metadata file missing: {METADATA_PATH}")
        obj = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        _assert(isinstance(obj, dict), "dataset_metadata.json must be a JSON object")

        # Required fields
        dataset_info = obj.get("dataset_info")
        embeddings_meta = obj.get("embeddings")
        graph_meta = obj.get("graph")

        _assert(isinstance(dataset_info, dict), "metadata['dataset_info'] missing or not an object")
        _assert(isinstance(embeddings_meta, dict), "metadata['embeddings'] missing or not an object")
        _assert(isinstance(graph_meta, dict), "metadata['graph'] missing or not an object")

        created_at = dataset_info.get("created_at")
        _assert(isinstance(created_at, str) and created_at.strip(), "metadata['dataset_info']['created_at'] missing")
        try:
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError(f"created_at is not ISO-8601 parseable: {created_at!r}") from e

        # Embeddings consistency
        model_name = embeddings_meta.get("model")
        _assert(isinstance(model_name, str) and model_name.strip(), "metadata['embeddings']['model'] missing")
        dim = embeddings_meta.get("dimension")
        _assert(isinstance(dim, int) and dim > 0, "metadata['embeddings']['dimension'] missing or invalid")
        emb_tensor = embeddings_obj.get("embeddings")
        _assert(torch.is_tensor(emb_tensor) and emb_tensor.ndim == 2, "Loaded embeddings tensor invalid")
        _assert(int(emb_tensor.shape[1]) == int(dim), f"Embedding dim mismatch: metadata={dim} tensor={int(emb_tensor.shape[1])}")

        artifact_model = embeddings_obj.get("model_name")
        if isinstance(artifact_model, str) and artifact_model.strip():
            _assert(
                artifact_model.strip() == model_name.strip(),
                f"Embedding model mismatch: metadata={model_name!r} artifact={artifact_model!r}",
            )

        # Graph consistency
        num_nodes = graph_meta.get("num_nodes")
        num_edges = graph_meta.get("num_edges")
        _assert(isinstance(num_nodes, int) and num_nodes > 0, "metadata['graph']['num_nodes'] missing or invalid")
        _assert(isinstance(num_edges, int) and num_edges > 0, "metadata['graph']['num_edges'] missing or invalid")
        _assert(num_nodes == G.number_of_nodes(), f"Graph node count mismatch: metadata={num_nodes} graph={G.number_of_nodes()}")
        _assert(num_edges == G.number_of_edges(), f"Graph edge count mismatch: metadata={num_edges} graph={G.number_of_edges()}")

        sim_thresh = graph_meta.get("similarity_threshold", None)
        if sim_thresh is not None:
            _assert(isinstance(sim_thresh, (int, float)), "metadata['graph']['similarity_threshold'] must be number or null")

        print("Dataset Version Info:")
        print(f"Model: {model_name}")
        print(f"Nodes: {num_nodes}")
        print(f"Edges: {num_edges}")
        print(f"Similarity Threshold: {sim_thresh}")
        print(f"Created At: {created_at}")

    results.append(_run_phase("DATASET METADATA VALIDATION", phase_metadata_validation))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_dataset_init() -> None:
        nonlocal dataset
        cls = _resolve_training_dataset_class()
        if cls is None:
            print("[info] No canonical training Dataset class found in repo.")
            print("[info] Using GraphEmbeddingDataset (built into this verifier) for batch-safety checks.")
            dataset = GraphEmbeddingDataset(embeddings_obj=embeddings_obj, G=G, k_neighbors=16)
        else:
            print(f"[info] Using repo Dataset class: {cls.__module__}.{cls.__name__}")
            dataset = cls()  # type: ignore[call-arg]

        _assert(dataset is not None, "Dataset failed to initialize")
        print(f"Dataset class: {dataset.__class__.__name__}")
        print(f"Total samples: {len(dataset):,}")
        _assert(len(dataset) > 0, "Dataset has zero length")
        _print_mem(prefix="After dataset init: ")

    results.append(_run_phase("DATASET IMPORT + INITIALIZATION", phase_dataset_init))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_sample_inspection() -> None:
        nonlocal sample0
        _assert(dataset is not None, "Dataset not initialized")
        sample0 = dataset[0]  # type: ignore[index]
        _assert(sample0 is not None, "dataset[0] returned None")
        _assert(isinstance(sample0, dict), f"dataset[0] must return dict, got {type(sample0)}")
        print(f"Sample keys: {sorted(sample0.keys())}")

        for k, v in sample0.items():
            if v is None:
                raise ValueError(f"Sample contains None for key {k!r}")
            _describe_tensor(k, v)

        non_tensor = [k for k, v in sample0.items() if not torch.is_tensor(v)]
        _assert(len(non_tensor) == 0, f"Non-tensor sample values detected: {non_tensor}")

    results.append(_run_phase("SAMPLE STRUCTURAL INSPECTION", phase_sample_inspection))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_tensor_consistency() -> None:
        _assert(dataset is not None, "Dataset not initialized")
        _assert(sample0 is not None and isinstance(sample0, dict), "Need sample0")

        n = len(dataset)
        N = 10
        idxs = [0] + [random.randrange(0, n) for _ in range(N - 1)]

        ref = sample0
        ref_shapes = {k: tuple(v.shape) for k, v in ref.items() if torch.is_tensor(v)}
        ref_dtypes = {k: v.dtype for k, v in ref.items() if torch.is_tensor(v)}

        for i in idxs:
            s = dataset[i]  # type: ignore[index]
            _assert(isinstance(s, dict), f"dataset[{i}] returned non-dict: {type(s)}")
            _assert(set(s.keys()) == set(ref.keys()), f"Inconsistent keys at idx={i}")
            for k, v in s.items():
                _assert(torch.is_tensor(v), f"Non-tensor value at idx={i} key={k!r}: {type(v)}")
                _assert(tuple(v.shape) == ref_shapes[k], f"Shape mismatch at idx={i} key={k}: {tuple(v.shape)} vs {ref_shapes[k]}")
                _assert(v.dtype == ref_dtypes[k], f"Dtype mismatch at idx={i} key={k}: {v.dtype} vs {ref_dtypes[k]}")

        print("✓ feature dimension stable")
        print("✓ dtype stable")
        print("✓ graph structure consistent")

    results.append(_run_phase("TENSOR CONSISTENCY SCAN", phase_tensor_consistency))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_dataloader_batching() -> None:
        nonlocal batch0
        _assert(dataset is not None, "Dataset not initialized")
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        it = iter(loader)
        batch0 = next(it)
        _assert(isinstance(batch0, dict), f"Batch must be dict, got {type(batch0)}")
        print(f"Batch keys: {sorted(batch0.keys())}")
        for k, v in batch0.items():
            _describe_tensor(k, v)

        # Effective batch size check
        any_key = next(iter(batch0.keys()))
        bs = int(batch0[any_key].shape[0]) if torch.is_tensor(batch0[any_key]) else -1
        _assert(bs == 8, f"Effective batch size mismatch: expected 8, got {bs}")
        print(f"Effective batch size: {bs}")

    results.append(_run_phase("DATALOADER BATCHING TEST", phase_dataloader_batching))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_iteration_stress() -> None:
        _assert(dataset is not None, "Dataset not initialized")
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        t0 = time.time()
        batches = 0
        for i, _ in enumerate(loader):
            batches += 1
            if i + 1 >= 50:
                break
        dt = time.time() - t0
        bps = batches / max(dt, 1e-9)
        print(f"Iterated {batches} batches in {dt:.2f}s ⇒ ~{bps:.2f} batches/sec")
        _print_mem(prefix="After stress test: ")

    results.append(_run_phase("ITERATION STRESS TEST", phase_iteration_stress))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_dummy_forward() -> None:
        _assert(batch0 is not None and isinstance(batch0, dict), "Need a batch from DataLoader")
        _assert("node_features" in batch0, "Batch missing required key 'node_features'")
        feats = batch0["node_features"]
        _assert(torch.is_tensor(feats), "node_features must be tensor")
        _assert(feats.ndim == 2, f"Expected node_features to be [B, D], got shape {tuple(feats.shape)}")
        feature_dim = int(feats.shape[1])

        model = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        model.train()
        feats = feats.float()

        out = model(feats)
        _assert(out.shape[0] == feats.shape[0], "Output batch dim mismatch")
        loss = out.mean()
        loss.backward()
        print("✓ forward pass successful")
        print("✓ backward pass successful")

        # Basic gradient sanity
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            grad_norm += float(p.grad.detach().abs().mean())
        _assert(math.isfinite(grad_norm) and grad_norm > 0.0, "Gradients appear invalid (zero or non-finite)")
        print(f"Mean abs grad (sanity): {grad_norm:.6f}")

    results.append(_run_phase("DUMMY FORWARD-PASS VALIDATION", phase_dummy_forward))
    if not results[-1].ok:
        _final_report(results)
        raise SystemExit(1)

    def phase_reload_stability() -> None:
        nonlocal dataset
        _assert(dataset is not None, "Dataset not initialized")
        n0 = len(dataset)
        s0 = dataset[0]  # type: ignore[index]
        _assert(isinstance(s0, dict), "dataset[0] must be dict")
        keys0 = set(s0.keys())
        shapes0 = {k: tuple(v.shape) for k, v in s0.items() if torch.is_tensor(v)}
        dtypes0 = {k: v.dtype for k, v in s0.items() if torch.is_tensor(v)}

        del dataset
        dataset = None
        gc.collect()

        dataset2 = GraphEmbeddingDataset(embeddings_obj=embeddings_obj, G=G, k_neighbors=16)
        n1 = len(dataset2)
        _assert(n0 == n1, f"Dataset length changed after reload: {n0} vs {n1}")
        s1 = dataset2[0]
        _assert(isinstance(s1, dict), "dataset[0] must be dict after reload")
        _assert(set(s1.keys()) == keys0, "Sample keys changed after reload")
        for k, v in s1.items():
            _assert(torch.is_tensor(v), f"Non-tensor after reload key={k}")
            _assert(tuple(v.shape) == shapes0[k], f"Shape changed after reload key={k}")
            _assert(v.dtype == dtypes0[k], f"Dtype changed after reload key={k}")
        print("✓ reload stability confirmed (length + sample structure unchanged)")

    results.append(_run_phase("DATASET RELOAD STABILITY TEST", phase_reload_stability))

    _final_report(results)
    if not all(r.ok for r in results):
        raise SystemExit(1)


def _final_report(results: Sequence[PhaseResult]) -> None:
    print("\n" + "=" * 36)
    print("TRAINING READINESS SUMMARY")
    print("=" * 26)

    name_map = {
        "ARTIFACT DISCOVERY": "Artifacts",
        "DATASET LOADING": "Dataset Load",
        "STRUCTURAL VALIDATION": "Graph Artifacts",
        "DATASET METADATA VALIDATION": "Metadata",
        "DATASET IMPORT + INITIALIZATION": "Dataset Init",
        "SAMPLE STRUCTURAL INSPECTION": "Sample Structure",
        "TENSOR CONSISTENCY SCAN": "Shape Consistency",
        "DATALOADER BATCHING TEST": "Batching",
        "ITERATION STRESS TEST": "Iteration",
        "DUMMY FORWARD-PASS VALIDATION": "Forward Pass",
        "DATASET RELOAD STABILITY TEST": "Reload Stability",
    }

    ok_all = True
    for r in results:
        label = name_map.get(r.name, r.name)
        status = "PASS" if r.ok else "FAIL"
        ok_all = ok_all and r.ok
        print(f"{label:18s}: {status}")

    print("")
    print(f"FINAL STATUS: {'TRAINING READY' if ok_all else 'NOT READY'}")
    _print_mem(prefix="Final: ")


if __name__ == "__main__":
    main()

