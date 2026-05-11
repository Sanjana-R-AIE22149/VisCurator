/**
 * VisCurator — Copilot API Client
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Async generator that streams SSE chunks from the copilot/analyze
 * and builder/compile endpoints.
 */

import { BASE_URL, authHeaders } from './api';

// ── Types ───────────────────────────────────────────────────

interface GraphNode {
  id: string;
  type?: string;
  data: Record<string, unknown>;
  position: { x: number; y: number };
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  [key: string]: unknown;
}

// ── SSE Helpers ─────────────────────────────────────────────

/**
 * Read an SSE stream from a fetch Response and yield text chunks.
 */
async function* readSSEStream(response: Response): AsyncGenerator<string> {
  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      // Keep last (possibly incomplete) line in the buffer
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;

        const payload = trimmed.slice(6); // strip "data: "
        if (payload === '[DONE]') return;

        try {
          const parsed = JSON.parse(payload);
          if (parsed.error) {
            console.error('[Copilot SSE] Server error:', parsed.error);
            return;
          }
          if (parsed.chunk) {
            yield parsed.chunk as string;
          }
        } catch {
          // Non-JSON data line, skip
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// ── Copilot Analysis ────────────────────────────────────────

/**
 * Stream an architecture analysis from the backend copilot.
 *
 * Yields text chunks as they arrive from NVIDIA NIM.
 * Throws if the backend is unreachable.
 */
export async function* streamCopilotAnalysis(
  nodes: GraphNode[],
  edges: GraphEdge[],
  signal?: AbortSignal
): AsyncGenerator<string> {
  const serializedNodes = nodes.map((n) => ({
    id: n.id,
    type: n.type,
    data: n.data,
    position: n.position,
  }));

  const serializedEdges = edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
  }));

  const res = await fetch(`${BASE_URL}/api/copilot/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ nodes: serializedNodes, edges: serializedEdges }),
    signal,
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Copilot API error ${res.status}: ${body}`);
  }

  yield* readSSEStream(res);
}

// ── Builder Compile ─────────────────────────────────────────

export interface CompileResult {
  code: string;
  model_summary: string;
  warnings: string[];
}

/**
 * Compile the React Flow graph into PyTorch code via the deterministic
 * code generator (no LLM needed).
 */
export async function compileGraph(
  nodes: GraphNode[],
  edges: GraphEdge[]
): Promise<CompileResult> {
  const serializedNodes = nodes.map((n) => ({
    id: n.id,
    type: n.type,
    data: n.data,
    position: n.position,
  }));

  const serializedEdges = edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
  }));

  const res = await fetch(`${BASE_URL}/api/builder/compile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ nodes: serializedNodes, edges: serializedEdges }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Compile API error ${res.status}: ${body}`);
  }

  return res.json();
}

