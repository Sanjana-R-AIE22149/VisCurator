/**
 * VisCurator — Backend API Client
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Thin wrappers around the FastAPI REST + WebSocket endpoints.
 * All functions are safe to call even when the backend is offline —
 * callers should catch errors and fall back to mock behaviour.
 */

// ── URLs ────────────────────────────────────────────────────

export const BASE_URL = 'http://localhost:8000';
export const WS_URL = 'ws://localhost:8000';

// ── Types ───────────────────────────────────────────────────

export interface PipelineMessage {
  type: 'thought' | 'tool_call' | 'tool_result' | 'log' | 'script_log' | 'done' | 'error';
  message: string;
  data: Record<string, unknown>;
  timestamp?: string;
}

export interface DatasetSearchResponse {
  job_id: string;
  status: string;
  message: string;
  created_at: string;
}

// ── Map frontend source labels → backend enum values ────────

const SOURCE_MAP: Record<string, string> = {
  HuggingFace: 'huggingface',
  OpenImages: 'openimages',
};

// ── REST: POST /api/dataset/search ──────────────────────────

export async function startDatasetPipeline(
  query: string,
  source: string,
  targetSize: number
): Promise<{ job_id: string }> {
  const res = await fetch(`${BASE_URL}/api/dataset/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      source: SOURCE_MAP[source] ?? source.toLowerCase(),
      target_size: targetSize,
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Backend returned ${res.status}: ${body}`);
  }

  const data: DatasetSearchResponse = await res.json();
  return { job_id: data.job_id };
}

// ── WebSocket: /ws/pipeline/{job_id} ────────────────────────

export function connectPipelineWebSocket(
  jobId: string,
  onMessage: (msg: PipelineMessage) => void,
  onClose: () => void
): WebSocket {
  const ws = new WebSocket(`${WS_URL}/ws/pipeline/${jobId}`);

  ws.onmessage = (event) => {
    try {
      const parsed: PipelineMessage = JSON.parse(event.data);
      onMessage(parsed);
    } catch {
      console.warn('[WS] Failed to parse message:', event.data);
    }
  };

  ws.onclose = () => {
    onClose();
  };

  ws.onerror = (err) => {
    console.error('[WS] Connection error:', err);
  };

  return ws;
}

// ── REST: GET /api/health ───────────────────────────────────

export async function getHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/api/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}
