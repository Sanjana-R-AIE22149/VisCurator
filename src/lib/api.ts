/**
 * VisCurator — Backend API Client
 */

export const BASE_URL = 'http://localhost:8000';
export const WS_URL   = 'ws://localhost:8000';

// ── Token storage ─────────────────────────────────────────────────────────────

const TOKEN_KEY = 'vc_token';

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getToken();
  return token
    ? { Authorization: `Bearer ${token}`, ...extra }
    : { ...extra };
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export interface LoginResponse {
  access_token: string;
  token_type: string;
  username: string;
  name: string;
  role: string;
}

export async function apiLogin(username: string, password: string): Promise<LoginResponse> {
  const body = new URLSearchParams({ username, password });
  const res = await fetch(`${BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error((detail as { detail?: string }).detail ?? `Login failed (${res.status})`);
  }
  return res.json();
}

// ── Shared types ──────────────────────────────────────────────────────────────

export interface PipelineMessage {
  type: 'thought' | 'tool_call' | 'tool_result' | 'log' | 'script_log' | 'done' | 'error';
  message: string;
  data: Record<string, unknown>;
  timestamp?: string;
}

export interface DatasetOption {
  source: string;
  dataset_id: string;
  name: string;
  description: string;
  size_estimate: string;
  url: string;
  pros: string[];
  cons: string[];
}

export interface DatasetSearchResponse {
  job_id: string;
  status: string;
  message: string;
  created_at: string;
}

export interface JobListItem {
  job_id: string;
  status: string;
  query: string;
  target_size: number;
  created_at: string;
  summary: string;
  paused: boolean;
}

export interface BuilderTrainResponse {
  run_id: string;
  status: string;
  task_type: 'mnist_classification' | 'object_detection';
}

export interface TrainingMetricPoint {
  run_id: string;
  epoch: number;
  loss: number;
  accuracy: number;
  precision: number;
  recall: number;
  map: number;
  timestamp: string;
}

export interface TrainingRunSummary {
  run_id: string;
  epochs_recorded: number;
  started_at: string;
  updated_at: string;
  best_loss: number;
  best_accuracy: number;
  best_map: number;
  status?: string;
  task_type?: string;
}

const SOURCE_MAP: Record<string, string> = {
  HuggingFace: 'huggingface',
  OpenImages:  'openimages',
};

// ── Dataset ───────────────────────────────────────────────────────────────────

export async function startDatasetPipeline(
  query: string,
  source: string,
  targetSize: number,
): Promise<{ job_id: string }> {
  const res = await fetch(`${BASE_URL}/api/dataset/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ query, source: SOURCE_MAP[source] ?? 'huggingface', target_size: targetSize }),
  });
  if (!res.ok) throw new Error(`Backend returned ${res.status}`);
  const data: DatasetSearchResponse = await res.json();
  return { job_id: data.job_id };
}

export async function replyToJob(jobId: string, reply: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/api/dataset/reply/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ reply }),
  });
  if (!res.ok) throw new Error(`Reply API returned ${res.status}`);
}

export async function getDatasetJobs(): Promise<JobListItem[]> {
  try {
    const res = await fetch(`${BASE_URL}/api/dataset/jobs`, {
      headers: authHeaders(),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

// ── WebSockets (token passed as query param since WS headers aren't standard) ─

export function connectPipelineWebSocket(
  jobId: string,
  onMessage: (msg: PipelineMessage) => void,
  onClose: (event?: CloseEvent) => void,
  onError?: (event: Event) => void,
): WebSocket {
  const token = getToken();
  const url = token
    ? `${WS_URL}/ws/pipeline/${jobId}?token=${encodeURIComponent(token)}`
    : `${WS_URL}/ws/pipeline/${jobId}`;
  const ws = new WebSocket(url);
  ws.onmessage = (event) => {
    try { onMessage(JSON.parse(event.data) as PipelineMessage); }
    catch { console.warn('[WS] Failed to parse message:', event.data); }
  };
  ws.onclose  = (event) => onClose(event);
  ws.onerror  = (err)   => { console.error('[WS] Error:', err); onError?.(err); };
  return ws;
}

export function connectTrainingWebSocket(
  runId: string,
  onMessage: (msg: PipelineMessage) => void,
  onClose: (event?: CloseEvent) => void,
  onError?: (event: Event) => void,
): WebSocket {
  const token = getToken();
  const url = token
    ? `${WS_URL}/ws/train/${runId}?token=${encodeURIComponent(token)}`
    : `${WS_URL}/ws/train/${runId}`;
  const ws = new WebSocket(url);
  ws.onmessage = (event) => {
    try { onMessage(JSON.parse(event.data) as PipelineMessage); }
    catch { console.warn('[Train WS] Failed to parse message:', event.data); }
  };
  ws.onclose  = (event) => onClose(event);
  ws.onerror  = (err)   => { console.error('[Train WS] Error:', err); onError?.(err); };
  return ws;
}

// ── Health (public — no auth needed) ─────────────────────────────────────────

export async function getHealth(): Promise<{ online: boolean; nim_connected: boolean }> {
  try {
    const res = await fetch(`${BASE_URL}/api/health`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return { online: false, nim_connected: false };
    const data = await res.json();
    return { online: true, nim_connected: data.nim_connected ?? false };
  } catch {
    return { online: false, nim_connected: false };
  }
}

export const fetchHealth = getHealth;

// ── Builder ───────────────────────────────────────────────────────────────────

export async function startTrainingRun(
  nodes: unknown[],
  edges: unknown[],
  taskType: 'mnist_classification' | 'object_detection',
  datasetPath?: string | null,
): Promise<BuilderTrainResponse> {
  const res = await fetch(`${BASE_URL}/api/builder/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ nodes, edges, task_type: taskType, dataset_path: datasetPath ?? null }),
  });
  if (!res.ok) throw new Error(`Training API returned ${res.status}`);
  return res.json();
}

export async function getTrainingRuns(): Promise<TrainingRunSummary[]> {
  const res = await fetch(`${BASE_URL}/api/builder/train/runs`, {
    headers: authHeaders(),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`Training runs API returned ${res.status}`);
  return res.json();
}

export async function getTrainingMetrics(runId: string): Promise<TrainingMetricPoint[]> {
  const res = await fetch(`${BASE_URL}/api/builder/train/${runId}/metrics`, {
    headers: authHeaders(),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`Training metrics API returned ${res.status}`);
  return res.json();
}
