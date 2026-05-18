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

export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
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

async function apiFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const res = await fetch(url, options);
  if (res.status === 401) {
    clearToken();
    window.location.href = '/login';
    throw new Error('Session expired. Please log in again.');
  }
  return res;
}

// ── Shared types ──────────────────────────────────────────────────────────────

export interface PipelineMessage {
  id: string;
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
  console.log('[API REQ] startDatasetPipeline:', { query, source, targetSize });
  const res = await apiFetch(`${BASE_URL}/api/dataset/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ query, source: SOURCE_MAP[source] ?? 'huggingface', target_size: targetSize }),
  });
  if (!res.ok) {
    console.error('[API RES] startDatasetPipeline FAIL:', res.status);
    throw new Error(`Backend returned ${res.status}`);
  }
  const data: DatasetSearchResponse = await res.json();
  console.log('[API RES] startDatasetPipeline OK:', data);
  return { job_id: data.job_id };
}

export async function replyToJob(jobId: string, reply: string): Promise<void> {
  console.log('[API REQ] replyToJob:', { jobId, reply });
  const res = await apiFetch(`${BASE_URL}/api/dataset/reply/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ reply }),
  });
  if (!res.ok) {
    console.error('[API RES] replyToJob FAIL:', res.status);
    throw new Error(`Reply API returned ${res.status}`);
  }
  console.log('[API RES] replyToJob OK');
}

export interface DatasetInfo {
  id: string;
  path: string;
  name: string;
  metadata?: {
    resolution?: string;
    num_classes?: number;
    image_count?: number;
  };
}

export async function getLocalDatasets(): Promise<DatasetInfo[]> {
  const res = await apiFetch(`${BASE_URL}/api/dataset/list`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export interface DatasetInspection {
  path: string;
  num_classes: number;
  class_names: string[];
  total_images: number;
  sample_width: number;
  sample_height: number;
  resolution: string;
}

export async function inspectDataset(path: string): Promise<DatasetInspection> {
  const res = await apiFetch(`${BASE_URL}/api/dataset/inspect?path=${encodeURIComponent(path)}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error('Failed to inspect dataset');
  return res.json();
}

export async function uploadDataset(file: File): Promise<{ job_id: string; files: string[]; slug: string }> {
  console.log('[API REQ] uploadDataset:', file.name);
  const formData = new FormData();
  formData.append('file', file);
  
  const res = await apiFetch(`${BASE_URL}/api/dataset/upload`, {
    method: 'POST',
    headers: authHeaders(), // FormData sets its own Content-Type with boundary
    body: formData,
  });
  
  if (!res.ok) {
    console.error('[API RES] uploadDataset FAIL:', res.status);
    throw new Error(`Backend returned ${res.status}`);
  }
  
  const data = await res.json();
  console.log('[API RES] uploadDataset OK:', data);
  
  // Fetch job status to get the file list
  const statusRes = await apiFetch(`${BASE_URL}/api/dataset/status/${data.job_id}`, { headers: authHeaders() });
  const statusData = await statusRes.json();
  
  return { 
    job_id: data.job_id, 
    files: statusData.result?.files || [],
    slug: statusData.result?.dataset_id || ''
  };
}

export async function uploadSeedClass(jobId: string, className: string, files: FileList | File[]): Promise<void> {
  console.log('[API REQ] uploadSeedClass:', { jobId, className, fileCount: files.length });
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }
  
  const res = await apiFetch(`${BASE_URL}/api/dataset/seed/${jobId}/${className}`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  });
  
  if (!res.ok) {
    console.error('[API RES] uploadSeedClass FAIL:', res.status);
    throw new Error(`Backend returned ${res.status}`);
  }
  console.log('[API RES] uploadSeedClass OK');
}

export async function startAnnotation(
  jobId: string,
  minConfidence = 0.30,
  blurThreshold = 80.0,
): Promise<void> {
  console.log('[API REQ] startAnnotation:', { jobId, minConfidence, blurThreshold });
  const res = await apiFetch(`${BASE_URL}/api/dataset/annotate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      job_id:          jobId,
      seeds:           {},
      min_confidence:  minConfidence,
      blur_threshold:  blurThreshold,
    }),
  });

  if (!res.ok) {
    console.error('[API RES] startAnnotation FAIL:', res.status);
    let detail = '';
    try {
      const errData = await res.json();
      detail = errData.detail || errData.message || '';
    } catch {
      // ignore
    }
    throw new Error(`Backend returned ${res.status}${detail ? ': ' + detail : ''}`);
  }
}

export interface AugmentOptions {
  nAug?: number;
  targetSize?: number;
}

export interface AnnotatedSample {
  url: string;
  label: string;
  confidence: number;
  confidence_band: 'high' | 'medium' | 'low';
}

export interface AnnotationReport {
  job_id: string;
  classes: string[];
  min_confidence: number;
  class_counts: Record<string, number>;
  confidence_distribution: { high: number; medium: number; low: number };
  low_confidence_count: number;
  annotated_samples: AnnotatedSample[];
  low_confidence_samples: AnnotatedSample[];
}

export async function getAnnotationReport(jobId: string): Promise<AnnotationReport> {
  const res = await apiFetch(`${BASE_URL}/api/dataset/annotation-report/${jobId}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Annotation report not found (${res.status})`);
  return res.json();
}

/**
 * Triggers a browser download of the augmented dataset ZIP.
 * Uses a direct fetch + blob URL so the token is sent as a header
 * (avoids leaking it in the URL bar).
 */
export async function downloadAugmentedDataset(jobId: string): Promise<void> {
  console.log('[API REQ] downloadAugmentedDataset:', { jobId });
  const res = await apiFetch(`${BASE_URL}/api/dataset/download-augmented/${jobId}`, {
    headers: authHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `Download failed (${res.status})`);
  }
  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `augmented_${jobId.slice(0, 8)}.zip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function startAugmentation(jobId: string, options: AugmentOptions = {}): Promise<{ out_dir: string }> {
  console.log('[API REQ] startAugmentation:', { jobId, options });
  const res = await apiFetch(`${BASE_URL}/api/dataset/augment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      job_id:      jobId,
      n_aug:       options.nAug      ?? 4,
      target_size: options.targetSize ?? 224,
    }),
  });

  if (!res.ok) {
    console.error('[API RES] startAugmentation FAIL:', res.status);
    let detail = '';
    try {
      const errData = await res.json();
      detail = errData.detail || errData.message || '';
    } catch { /* ignore */ }
    throw new Error(`Backend returned ${res.status}${detail ? ': ' + detail : ''}`);
  }
  const data = await res.json();
  console.log('[API RES] startAugmentation OK:', data);
  return data;
}

export async function getDatasetJobs(): Promise<JobListItem[]> {
  try {
    const res = await apiFetch(`${BASE_URL}/api/dataset/jobs`, {
      headers: authHeaders(),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

// ── WebSocket Management ──────────────────────────────────────────────────────

type WebSocketCallback = (msg: PipelineMessage) => void;

class SafeWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private onMessage: WebSocketCallback;
  private onClose: (event?: CloseEvent) => void;
  private onError?: (event: Event) => void;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private pingInterval: number | null = null;
  private intentionallyClosed = false;

  constructor(
    url: string,
    onMessage: WebSocketCallback,
    onClose: (event?: CloseEvent) => void,
    onError?: (event: Event) => void
  ) {
    this.url = url;
    this.onMessage = onMessage;
    this.onClose = onClose;
    this.onError = onError;
    this.connect();
  }

  private connect() {
    const token = getToken();
    const finalUrl = token ? `${this.url}${this.url.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}` : this.url;
    
    this.ws = new WebSocket(finalUrl);
    this.intentionallyClosed = false;

    this.ws.onopen = () => {
      console.log(`[WS] Connected to ${this.url}`);
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      this.startHeartbeat();
    };

    this.ws.onmessage = (event) => {
      if (event.data === 'pong' || event.data === '{"type":"pong"}') return;
      try {
        const msg = JSON.parse(event.data) as PipelineMessage;
        this.onMessage(msg);
      } catch (err) {
        console.warn('[WS] Failed to parse message:', event.data, err);
      }
    };

    this.ws.onclose = (event) => {
      this.stopHeartbeat();
      if (this.intentionallyClosed) {
        this.onClose(event);
        return;
      }

      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})...`);
        setTimeout(() => this.connect(), delay);
      } else {
        console.error('[WS] Max reconnect attempts reached');
        this.onClose(event);
      }
    };

    this.ws.onerror = (err) => {
      console.error('[WS] Error:', err);
      this.onError?.(err);
    };
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.pingInterval = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 15000);
  }

  private stopHeartbeat() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  public send(data: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    }
  }

  public close() {
    this.intentionallyClosed = true;
    this.stopHeartbeat();
    this.ws?.close();
  }
}

export function connectPipelineWebSocket(
  jobId: string,
  onMessage: (msg: PipelineMessage) => void,
  onClose: (event?: CloseEvent) => void,
  onError?: (event: Event) => void,
): SafeWebSocket {
  return new SafeWebSocket(`${WS_URL}/ws/pipeline/${jobId}`, onMessage, onClose, onError);
}

export function connectTrainingWebSocket(
  runId: string,
  onMessage: (msg: PipelineMessage) => void,
  onClose: (event?: CloseEvent) => void,
  onError?: (event: Event) => void,
): SafeWebSocket {
  return new SafeWebSocket(`${WS_URL}/ws/train/${runId}`, onMessage, onClose, onError);
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
  taskType: 'mnist_classification' | 'object_detection' | 'custom_curated',
  datasetPath?: string | null,
  epochs?: number,
  numImages?: number,
): Promise<BuilderTrainResponse> {
  const res = await apiFetch(`${BASE_URL}/api/builder/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ nodes, edges, task_type: taskType, dataset_path: datasetPath ?? null, epochs, num_images: numImages }),
  });
  if (!res.ok) throw new Error(`Training API returned ${res.status}`);
  return res.json();
}

export async function getTrainingRuns(): Promise<TrainingRunSummary[]> {
  const res = await apiFetch(`${BASE_URL}/api/builder/train/runs`, {
    headers: authHeaders(),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`Training runs API returned ${res.status}`);
  return res.json();
}

export async function getTrainingMetrics(runId: string): Promise<TrainingMetricPoint[]> {
  const res = await apiFetch(`${BASE_URL}/api/builder/train/${runId}/metrics`, {
    headers: authHeaders(),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`Training metrics API returned ${res.status}`);
  return res.json();
}
