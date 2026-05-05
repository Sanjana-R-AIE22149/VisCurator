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
        warnings: list[str] = []

        # ── 1. Build adjacency graph ──
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {}
        node_map: dict[str, dict[str, Any]] = {}

        for n in nodes:
            nid = n["id"]
            node_map[nid] = n
            in_degree.setdefault(nid, 0)

        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            if src in node_map and tgt in node_map:
                adj[src].append(tgt)
                in_degree[tgt] = in_degree.get(tgt, 0) + 1

        # ── 2. Topological sort (Kahn's algorithm) ──
        queue: deque[str] = deque()
        for nid, deg in in_degree.items():
            if deg == 0:
                queue.append(nid)

        topo_order: list[str] = []
        while queue:
            nid = queue.popleft()
            topo_order.append(nid)
            for child in adj.get(nid, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Detect disconnected nodes
        connected_ids = set(topo_order)
        for n in nodes:
            if n["id"] not in connected_ids:
                warnings.append(
                    f"Node '{n.get('data', {}).get('label', n['id'])}' "
                    f"is disconnected from the graph."
                )
                topo_order.append(n["id"])  # still include it

        # ── 3. Classify nodes & gather layer specs ──
        layers: list[dict[str, Any]] = []
        has_input = False
        input_channels = 3
        input_h, input_w = 224, 224

        for nid in topo_order:
            node = node_map[nid]
            data = node.get("data", {})
            kind = _classify_node(node)
            var_name = _sanitize_id(nid)

            spec: dict[str, Any] = {
                "id": nid,
                "var": var_name,
                "kind": kind,
                "label": data.get("label", kind),
                "data": data,
            }

            if kind == "input":
                has_input = True
                input_channels = _safe_int(data.get("channels"), 3)
                input_h, input_w = _parse_resolution(data.get("resolution", "224×224"))
                spec["channels"] = input_channels
                spec["resolution"] = (input_h, input_w)

            elif kind == "conv":
                filters = _safe_int(data.get("filters"), 64)
                kernel = _parse_kernel(data.get("kernel", 3))
                act = str(data.get("activation", "ReLU"))
                spec["filters"] = filters
                spec["kernel"] = kernel
                spec["activation"] = act

            elif kind == "batchnorm":
                spec["num_features"] = _safe_int(data.get("filters"), 64)

            elif kind == "pooling":
                spec["kernel"] = _parse_kernel(data.get("kernel", 2))

            elif kind == "linear":
                spec["out_features"] = _safe_int(data.get("filters"), 512)
                act = str(data.get("activation", "ReLU"))
                spec["activation"] = act

            elif kind == "attention":
                heads = _safe_int(data.get("heads"), 8)
                dim_k = _safe_int(data.get("dimK"), 64)
                spec["heads"] = heads
                spec["dim_k"] = dim_k
                spec["embed_dim"] = heads * dim_k

            elif kind == "residual":
                spec["filters"] = _safe_int(data.get("filters"), 64)

            layers.append(spec)

        if not has_input:
            warnings.append(
                "No input node found. Assuming default input: 3×224×224."
            )

        # ── 4. Shape tracking & incompatibility detection ──
        cur_channels = input_channels
        cur_h, cur_w = input_h, input_w
        is_flat = False  # whether we've flattened for linear layers
        layer_summaries: list[str] = []
        param_count = 0

        init_lines: list[str] = []
        forward_lines: list[str] = []
        extra_classes: list[str] = []

        for spec in layers:
            kind = spec["kind"]
            var = spec["var"]

            if kind == "input":
                layer_summaries.append(
                    f"Input: {spec['channels']}×{spec['resolution'][0]}×{spec['resolution'][1]}"
                )
                forward_lines.append(
                    f"        # Input: ({spec['channels']}, {spec['resolution'][0]}, {spec['resolution'][1]})"
                )
                continue

            if kind == "conv":
                f = spec["filters"]
                k = spec["kernel"]
                pad = k // 2
                act = spec["activation"]

                # Conv + BN + Activation as a sequential
                sub_modules = [
                    f"nn.Conv2d({cur_channels}, {f}, kernel_size={k}, padding={pad})",
                    f"nn.BatchNorm2d({f})",
                ]
                act_code = _activation_code(act)
                if act_code:
                    sub_modules.append(act_code)

                init_lines.append(
                    f"        self.{var} = nn.Sequential(\n"
                    + "".join(f"            {m},\n" for m in sub_modules)
                    + "        )"
                )
                forward_lines.append(f"        x = self.{var}(x)")

                params = cur_channels * f * k * k + f  # conv weights + bias
                params += f * 2  # BN gamma + beta
                param_count += params
                layer_summaries.append(
                    f"Conv2d({cur_channels}→{f}, {k}×{k}) + BN + {act}"
                )

                cur_channels = f
                # Spatial dims unchanged with padding=k//2
                continue

            if kind == "batchnorm":
                nf = spec["num_features"]
                if nf != cur_channels:
                    warnings.append(
                        f"BatchNorm2d expects {nf} features but previous "
                        f"layer outputs {cur_channels} channels."
                    )
                    nf = cur_channels

                init_lines.append(
                    f"        self.{var} = nn.BatchNorm2d({nf})"
                )
                forward_lines.append(f"        x = self.{var}(x)")
                param_count += nf * 2
                layer_summaries.append(f"BatchNorm2d({nf})")
                continue

            if kind == "pooling":
                k = spec["kernel"]
                init_lines.append(
                    f"        self.{var} = nn.MaxPool2d(kernel_size={k}, stride={k})"
                )
                forward_lines.append(f"        x = self.{var}(x)")
                cur_h = max(1, cur_h // k)
                cur_w = max(1, cur_w // k)
                layer_summaries.append(f"MaxPool2d({k}×{k})")
                continue

            if kind == "linear":
                out_f = spec["out_features"]
                act = spec.get("activation", "—")

                if not is_flat:
                    in_features = cur_channels * cur_h * cur_w
                    forward_lines.append(
                        f"        x = x.flatten(1)  "
                        f"# ({cur_channels}, {cur_h}, {cur_w}) → {in_features}"
                    )
                    is_flat = True
                else:
                    in_features = cur_channels  # reuse last out_features

                sub = [f"nn.Linear({in_features}, {out_f})"]
                act_code = _activation_code(act)
                if act_code:
                    sub.append(act_code)

                init_lines.append(
                    f"        self.{var} = nn.Sequential(\n"
                    + "".join(f"            {m},\n" for m in sub)
                    + "        )"
                )
                forward_lines.append(f"        x = self.{var}(x)")

                params = in_features * out_f + out_f
                param_count += params
                layer_summaries.append(f"Linear({in_features}→{out_f})")
                cur_channels = out_f
                continue

            if kind == "attention":
                embed = spec["embed_dim"]
                heads = spec["heads"]

                if not is_flat:
                    # Flatten spatial dims and project to embed_dim
                    in_features = cur_channels * cur_h * cur_w
                    init_lines.append(
                        f"        self.{var}_proj = nn.Linear({in_features}, {embed})"
                    )
                    forward_lines.append(
                        f"        x = x.flatten(1)  "
                        f"# ({cur_channels}, {cur_h}, {cur_w}) → {in_features}"
                    )
                    forward_lines.append(
                        f"        x = self.{var}_proj(x)  # → {embed}"
                    )
                    param_count += in_features * embed + embed
                    is_flat = True
                elif cur_channels != embed:
                    init_lines.append(
                        f"        self.{var}_proj = nn.Linear({cur_channels}, {embed})"
                    )
                    forward_lines.append(
                        f"        x = self.{var}_proj(x)  # {cur_channels} → {embed}"
                    )
                    param_count += cur_channels * embed + embed

                init_lines.append(
                    f"        self.{var} = nn.MultiheadAttention(\n"
                    f"            embed_dim={embed}, num_heads={heads}, batch_first=True,\n"
                    f"        )"
                )
                # MHA expects (batch, seq_len, embed_dim) — treat as single-token sequence
                forward_lines.append(
                    f"        x = x.unsqueeze(1)  # (B, 1, {embed}) — single-token sequence"
                )
                forward_lines.append(
                    f"        x, _ = self.{var}(x, x, x)"
                )
                forward_lines.append(
                    f"        x = x.squeeze(1)  # (B, {embed})"
                )

                # MHA params: 3 * embed^2 (QKV projections) + out projection
                mha_params = 4 * embed * embed + 4 * embed
                param_count += mha_params
                layer_summaries.append(
                    f"MultiheadAttention(embed={embed}, heads={heads})"
                )
                cur_channels = embed
                continue

            if kind == "residual":
                f = spec["filters"]
                cls_name = f"ResidualBlock_{f}"
                if not any(cls_name in c for c in extra_classes):
                    extra_classes.append(
                        f"class {cls_name}(nn.Module):\n"
                        f"    \"\"\"Residual block with skip connection.\"\"\"\n\n"
                        f"    def __init__(self, channels: int = {f}):\n"
                        f"        super().__init__()\n"
                        f"        self.block = nn.Sequential(\n"
                        f"            nn.Conv2d(channels, channels, 3, padding=1),\n"
                        f"            nn.BatchNorm2d(channels),\n"
                        f"            nn.ReLU(inplace=True),\n"
                        f"            nn.Conv2d(channels, channels, 3, padding=1),\n"
                        f"            nn.BatchNorm2d(channels),\n"
                        f"        )\n"
                        f"        self.relu = nn.ReLU(inplace=True)\n\n"
                        f"    def forward(self, x):\n"
                        f"        return self.relu(self.block(x) + x)\n"
                    )
                if cur_channels != f:
                    warnings.append(
                        f"ResidualBlock expects {f} channels but input has {cur_channels}."
                    )
                init_lines.append(
                    f"        self.{var} = {cls_name}({f})"
                )
                forward_lines.append(f"        x = self.{var}(x)")
                res_params = 2 * (f * f * 9 + f) + 2 * (f * 2)
                param_count += res_params
                layer_summaries.append(f"ResidualBlock({f})")
                cur_channels = f
                continue

            # Unknown node type — emit a warning
            warnings.append(
                f"Unknown node type '{spec.get('label', kind)}' — skipped."
            )

        # ── 5. Assemble the final code ──
        output_dim = cur_channels

        code_parts: list[str] = [
            '"""',
            "CVAgent Model — Auto-generated by VisCurator",
            f"Estimated parameters: {param_count:,}",
            '"""',
            "",
            "import torch",
            "import torch.nn as nn",
            "",
        ]

        # Extra helper classes (e.g. ResidualBlock)
        for cls in extra_classes:
            code_parts.append("")
            code_parts.append(cls)
            code_parts.append("")

        # Main model class
        code_parts.append("")
        code_parts.append("class CVAgentModel(nn.Module):")
        code_parts.append(f'    """Auto-generated model with {len(layer_summaries)} layers."""')
        code_parts.append("")
        code_parts.append("    def __init__(self):")
        code_parts.append("        super().__init__()")

        if not init_lines:
            code_parts.append("        pass  # No layers defined")
        else:
            for line in init_lines:
                code_parts.append(line)

        code_parts.append("")
        code_parts.append("    def forward(self, x: torch.Tensor) -> torch.Tensor:")

        if not forward_lines:
            code_parts.append("        return x")
        else:
            for line in forward_lines:
                code_parts.append(line)
            code_parts.append("        return x")

        # Main block
        code_parts.append("")
        code_parts.append("")
        code_parts.append('if __name__ == "__main__":')
        code_parts.append("    model = CVAgentModel()")
        code_parts.append(f"    x = torch.randn(1, {input_channels}, {input_h}, {input_w})")
        code_parts.append("    out = model(x)")
        code_parts.append('    print(f"Output shape: {out.shape}")')
        code_parts.append(
            '    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")'
        )
        code_parts.append("")

        code = "\n".join(code_parts)

        # ── 6. Model summary ──
        summary_parts = [
            f"Layers: {len(layer_summaries)}",
            f"Estimated Parameters: {param_count:,}",
            f"Input shape: ({input_channels}, {input_h}, {input_w})",
            f"Output dimension: {output_dim}",
            "",
            "Layer order:",
        ]
        for i, ls in enumerate(layer_summaries, 1):
            summary_parts.append(f"  {i}. {ls}")

        model_summary = "\n".join(summary_parts)

        logger.info(
            "Code generated — %d layers, ~%s params, %d warnings",
            len(layer_summaries),
            f"{param_count:,}",
            len(warnings),
        )

        return {
            "code": code,
            "model_summary": model_summary,
            "warnings": warnings,
        }
