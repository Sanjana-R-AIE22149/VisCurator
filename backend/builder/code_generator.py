"""
VisCurator / CVAgent — PyTorch Code Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Deterministic compiler that converts a React Flow visual node graph
into a syntactically valid, runnable PyTorch ``nn.Module``.

No LLM is involved — this is pure graph analysis and template-based
code emission, guaranteeing reproducible output.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────

def _parse_kernel(raw: Any) -> int:
    """Extract an integer kernel size from strings like '3×3', '2x2', 3."""
    if isinstance(raw, int):
        return raw
    s = str(raw).replace("×", "x").replace("X", "x")
    match = re.search(r"(\d+)", s)
    return int(match.group(1)) if match else 3


def _parse_resolution(raw: Any) -> tuple[int, int]:
    """Extract (H, W) from strings like '224×224'."""
    s = str(raw).replace("×", "x").replace("X", "x")
    parts = re.findall(r"\d+", s)
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    if len(parts) == 1:
        v = int(parts[0])
        return v, v
    return 224, 224


def _safe_int(raw: Any, default: int = 64) -> int:
    """Coerce a value to int, falling back to *default*."""
    if isinstance(raw, int):
        return raw
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _sanitize_id(node_id: str) -> str:
    """Turn an arbitrary node ID into a valid Python identifier."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", node_id)
    if s and s[0].isdigit():
        s = "_" + s
    return s


def _activation_code(act: str) -> str | None:
    """Return nn.Module code for an activation name, or None."""
    act_lower = act.strip().lower()
    mapping = {
        "relu": "nn.ReLU(inplace=True)",
        "gelu": "nn.GELU()",
        "silu": "nn.SiLU(inplace=True)",
        "leakyrelu": "nn.LeakyReLU(0.1, inplace=True)",
        "sigmoid": "nn.Sigmoid()",
        "tanh": "nn.Tanh()",
    }
    return mapping.get(act_lower)


# ── Node Classification ─────────────────────────────────────

def _classify_node(node: dict[str, Any]) -> str:
    """Return a canonical layer kind from node type + data.label.

    The frontend reuses the ``convBlock`` node type for Conv2d,
    BatchNorm, MaxPool, and Linear — they're distinguished by
    ``data.label``.
    """
    ntype = node.get("type", "")
    label = str(node.get("data", {}).get("label", "")).lower()

    if ntype == "inputNode":
        return "input"
    if ntype == "attentionBlock":
        return "attention"
    if ntype == "residualBlock":
        return "residual"

    # convBlock subtypes
    if "batchnorm" in label:
        return "batchnorm"
    if "pool" in label:
        return "pooling"
    if "linear" in label or "fc" in label or "dense" in label:
        return "linear"
    if "conv" in label:
        return "conv"

    # Additional dedicated node types the user asked for
    if ntype == "linearNode":
        return "linear"
    if ntype == "poolingNode":
        return "pooling"
    if ntype == "batchNormNode":
        return "batchnorm"

    # Fallback — treat unknown convBlock variants as conv
    if ntype == "convBlock":
        return "conv"

    return "unknown"


# ── Code Generator ───────────────────────────────────────────

class PyTorchCodeGenerator:
    """Compile a React Flow node graph into a PyTorch nn.Module.

    Usage::

        gen = PyTorchCodeGenerator()
        result = gen.generate(nodes, edges)
        print(result["code"])
    """

    def generate(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return ``{"code": str, "model_summary": str, "warnings": list[str]}``."""
        try:
            return self._generate_internal(nodes, edges)
        except Exception as exc:
            logger.error("Graph compilation failed: %s", exc)
            return {
                "code": self.get_safe_default_model(),
                "model_summary": "Error: Graph compilation failed. Using safe fallback model.",
                "warnings": [f"Compilation error: {exc}"],
            }

    @staticmethod
    def get_safe_default_model() -> str:
        """Return a basic but valid PyTorch model as a last-resort fallback."""
        return """\
import torch
import torch.nn as nn

class CVAgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
    def forward(self, x):
        return self.net(x)
"""

    def _generate_internal(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        warnings: list[str] = []
        node_map = {n["id"]: n for n in nodes}
        adj = defaultdict(list)
        rev_adj = defaultdict(list)
        in_degree = {n["id"]: 0 for n in nodes}

        for e in edges:
            src, tgt = e.get("source"), e.get("target")
            if src in node_map and tgt in node_map:
                adj[src].append(tgt)
                rev_adj[tgt].append(src)
                in_degree[tgt] += 1

        # ── 1. Topological Sort & Cycle Detection ──
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        topo_order = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(topo_order) < len(nodes):
            unvisited = set(node_map.keys()) - set(topo_order)
            warnings.append(f"Graph cycle detected involving nodes: {list(unvisited)[:3]}. Some connections may be ignored.")
            topo_order.extend(list(unvisited))

        # ── 2. Shape Propagation & Layer Spec Gathering ──
        node_outputs = {}  # nid -> (channels, h, w)
        layers: list[dict[str, Any]] = []
        
        # Initial search for input node to establish baseline
        input_node_id = next((nid for nid in topo_order if _classify_node(node_map[nid]) == "input"), None)
        if not input_node_id:
            warnings.append("No input node found. Defaulting to 3x224x224.")
            cur_shape = (3, 224, 224)
        else:
            in_data = node_map[input_node_id].get("data", {})
            cur_shape = (
                _safe_int(in_data.get("channels"), 3),
                *_parse_resolution(in_data.get("resolution", "224x224"))
            )
            node_outputs[input_node_id] = cur_shape

        for nid in topo_order:
            node = node_map[nid]
            kind = _classify_node(node)
            data = node.get("data", {})
            
            # If node has predecessors, use the first one's output shape as input
            preds = rev_adj.get(nid, [])
            if preds and preds[0] in node_outputs:
                cur_shape = node_outputs[preds[0]]
            
            in_c, in_h, in_w = cur_shape
            spec: dict[str, Any] = {"id": nid, "var": _sanitize_id(nid), "kind": kind, "in_shape": cur_shape}

            if kind == "input":
                node_outputs[nid] = cur_shape
                continue # Input logic handled above

            elif kind == "conv":
                f = _safe_int(data.get("filters"), 64)
                k = _parse_kernel(data.get("kernel", 3))
                spec.update({"filters": f, "kernel": k, "activation": data.get("activation", "ReLU")})
                node_outputs[nid] = (f, in_h, in_w) # assuming padding=k//2

            elif kind == "batchnorm":
                # AUTO-FIX: Ensure features match incoming channels
                f = _safe_int(data.get("filters"), in_c)
                if f != in_c:
                    warnings.append(f"Auto-fixed BatchNorm '{nid}': adjusted {f} -> {in_c} channels.")
                    f = in_c
                spec["num_features"] = f
                node_outputs[nid] = (f, in_h, in_w)

            elif kind == "pooling":
                k = _parse_kernel(data.get("kernel", 2))
                spec["kernel"] = k
                node_outputs[nid] = (in_c, max(1, in_h // k), max(1, in_w // k))

            elif kind == "linear":
                out_f = _safe_int(data.get("filters"), 512)
                spec.update({"out_features": out_f, "activation": data.get("activation", "ReLU")})
                node_outputs[nid] = (out_f, 1, 1)

            elif kind == "attention":
                heads = _safe_int(data.get("heads"), 8)
                dim_k = _safe_int(data.get("dimK"), 64)
                embed = heads * dim_k
                spec.update({"heads": heads, "embed_dim": embed})
                node_outputs[nid] = (embed, 1, 1)

            elif kind == "residual":
                f = _safe_int(data.get("filters"), in_c)
                if f != in_c:
                    warnings.append(f"ResidualBlock '{nid}' input mismatch: expects {f}, got {in_c} channels.")
                spec["filters"] = f
                node_outputs[nid] = (f, in_h, in_w)

            else:
                node_outputs[nid] = cur_shape # pass-through for unknown
            
            layers.append(spec)

        # ── 3. Code Emission ──
        init_lines: list[str] = []
        forward_lines: list[str] = []
        extra_classes: list[str] = []
        is_flat = False
        param_count = 0

        for spec in layers:
            kind, var, in_shape = spec["kind"], spec["var"], spec["in_shape"]
            in_c, in_h, in_w = in_shape

            if kind == "conv":
                f, k, act = spec["filters"], spec["kernel"], spec["activation"]
                pad = k // 2
                act_m = _activation_code(act)
                init_lines.append(f"        self.{var} = nn.Sequential(nn.Conv2d({in_c}, {f}, {k}, padding={pad}), nn.BatchNorm2d({f}), {act_m})")
                forward_lines.append(f"        x = self.{var}(x)")
                param_count += (in_c * f * k * k + f) + (f * 2)

            elif kind == "batchnorm":
                nf = spec["num_features"]
                init_lines.append(f"        self.{var} = nn.BatchNorm2d({nf})")
                forward_lines.append(f"        x = self.{var}(x)")
                param_count += nf * 2

            elif kind == "pooling":
                k = spec["kernel"]
                init_lines.append(f"        self.{var} = nn.MaxPool2d({k}, {k})")
                forward_lines.append(f"        x = self.{var}(x)")

            elif kind == "linear":
                if not is_flat:
                    in_f = in_c * in_h * in_w
                    forward_lines.append(f"        x = x.flatten(1) # {in_c}x{in_h}x{in_w} -> {in_f}")
                    is_flat = True
                else:
                    in_f = in_c
                out_f, act = spec["out_features"], spec["activation"]
                act_m = _activation_code(act)
                init_lines.append(f"        self.{var} = nn.Sequential(nn.Linear({in_f}, {out_f}), {act_m})")
                forward_lines.append(f"        x = self.{var}(x)")
                param_count += (in_f * out_f + out_f)

            elif kind == "attention":
                embed, heads = spec["embed_dim"], spec["heads"]
                if not is_flat:
                    in_f = in_c * in_h * in_w
                    init_lines.append(f"        self.{var}_proj = nn.Linear({in_f}, {embed})")
                    forward_lines.append(f"        x = self.{var}_proj(x.flatten(1))")
                    is_flat = True
                init_lines.append(f"        self.{var} = nn.MultiheadAttention({embed}, {heads}, batch_first=True)")
                forward_lines.append(f"        x, _ = self.{var}(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1)); x = x.squeeze(1)")
                param_count += (4 * embed * embed + 4 * embed)

            elif kind == "residual":
                f = spec["filters"]
                cls_name = f"ResidualBlock_{f}"
                if not any(cls_name in c for c in extra_classes):
                    extra_classes.append(f"class {cls_name}(nn.Module):\\n    def __init__(self, c):\\n        super().__init__()\\n        self.b = nn.Sequential(nn.Conv2d(c,c,3,1,1), nn.BatchNorm2d(c), nn.ReLU(True), nn.Conv2d(c,c,3,1,1), nn.BatchNorm2d(c))\\n    def forward(self, x): return torch.relu(self.b(x) + x)")
                init_lines.append(f"        self.{var} = {cls_name}({f})")
                forward_lines.append(f"        x = self.{var}(x)")
                param_count += 2 * (f * f * 9 + f) + 2 * (f * 2)

        # ── 4. Assembly ──
        code = f"import torch\\nimport torch.nn as nn\\n\\n" + "\\n".join(extra_classes) + "\\n\\nclass CVAgentModel(nn.Module):\\n    def __init__(self):\\n        super().__init__()\\n" + "\\n".join(init_lines) + "\\n\\n    def forward(self, x):\\n" + "\\n".join(forward_lines) + "\\n        return x\\n"
        
        return {
            "code": code.replace("\\n", "\n"),
            "model_summary": f"Layers: {len(layers)}\\nParams: {param_count:,}\\nInput: {node_outputs.get(input_node_id, (3,224,224))}".replace("\\n", "\n"),
            "warnings": warnings,
        }
