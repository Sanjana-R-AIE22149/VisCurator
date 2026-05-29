import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react';

export interface AppUser {
  id: string;
  name: string;
  email: string;
  role: string;
  avatarInitials: string;
}

interface AuthState {
  isAuthenticated: boolean;
  user: AppUser | null;
}

export interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
}

interface UIState {
  sidebarOpen: boolean;
  toasts: Toast[];
  setupComplete: boolean;
  nimApiKey: string | null;
  shortcutHintDismissed: boolean;
}

export interface HardwareStats {
  gpuModel: string;
  vramUsed: number;
  vramTotal: number;
  cpuPercent: number;
  ramPercent: number;
}

export interface TerminalLog {
  id: string;
  timestamp: string;
  message: string;
  status: 'ok' | 'warn' | 'error' | 'info';
  msgType?: 'thought' | 'tool_call' | 'tool_result' | 'script_log' | 'log' | 'done' | 'error';
}

export interface DatasetResult {
  dataset_id: string;
  description: string;
  quality_score?: number;
  source: string;
  metadata?: Record<string, unknown>;
}

export interface ProcessingPlan {
  needs_blur_filtering: boolean;
  needs_deduplication: boolean;
  needs_augmentation: boolean;
  needs_synthetic_generation: boolean;
  recommended_augmentations: string[];
  recommended_model: string;
  reasoning: string;
  input_summary?: Record<string, unknown>;
}

export interface PreprocessingReport {
  dataset_id: string;
  job_id?: string;
  hf_pipeline?: boolean;
  output_dir?: string;
  plan?: ProcessingPlan;
  before_stats?: {
    images?: number;
    class_distribution?: Record<string, number>;
  };
  after_stats?: {
    images?: number;
    blur_filtered?: number;
    duplicates_removed?: number;
    class_distribution?: Record<string, number>;
  };
  class_distribution?: Record<string, number>;
  augmentation_summary?: {
    operations?: string[];
    augmented_images?: number;
    synthetic_generation_recommended?: boolean;
  };
  annotation_summary?: {
    available?: boolean;
    count?: number;
    generator?: string | null;
    error?: string | null;
  };
  deblur_summary?: {
    recovered_for_export?: number;
    avg_before?: number;
    avg_after?: number;
  };
  deblur_preview?: Array<{
    id: number;
    label: string;
    before_url: string;
    after_url: string;
    before_blur: number;
    after_blur: number;
  }>;
  blur_scatter?: Array<{ id: number; laplacian: number; resolution: number; accepted: boolean }>;
  stage_samples?: {
    raw: Array<{ url: string; label: string; id: number }>;
    filtered: Array<{ url: string; label: string; reason: string; id: number }>;
    processed: Array<{ url: string; label: string; id: number }>;
    recovered?: Array<{
      id: number;
      label: string;
      before_url: string;
      after_url: string;
      before_blur: number;
      after_blur: number;
    }>;
  };
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

export interface PipelineQuestion {
  type: 'clarification' | 'dataset_selection';
  question: string;
  options: DatasetOption[] | string[];
  recommendation?: string;
  context?: string;
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
  eta?: string;
  total_epochs?: number;
  task_type?: string;
}

interface DatasetState {
  isProcessing: boolean;
  isPaused: boolean;
  pendingQuestion: PipelineQuestion | null;
  localDataset: { slug: string; path: string; files: string[] } | null;
  setLocalDataset: (data: { slug: string; path: string; files: string[] } | null) => void;
  terminalLogs: TerminalLog[];
  datasetQuery: string;
  datasetSource: 'HuggingFace' | 'OpenImages';
  targetSize: number;
  blurData: Array<{ id: number; laplacian: number; resolution: number; accepted: boolean }>;
  jobId: string | null;
  datasetResults: DatasetResult[];
  // Live stats from quality tool
  imagesIngested: number | null;
  qualityScore: number | null;
  classBalance: string | null;
  duplicatePercentage: number | null;
  processingPlan: ProcessingPlan | null;
  preprocessingReport: PreprocessingReport | null;
  clonedRecipe: string | null;
}

interface TrainingState {
  isTraining: boolean;
  trainingRunId: string | null;
  trainingTaskType: 'mnist_classification' | 'object_detection' | 'custom_curated';
  trainingLogs: TerminalLog[];
  trainingMetrics: TrainingMetricPoint[];
}

interface FlowState {
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  updateNodeData: (nodeId: string, data: Record<string, any>) => void;
}

interface CopilotState {
  copilotResponse: string;
  isCopilotStreaming: boolean;
}

interface AppStore extends AuthState, UIState, DatasetState, TrainingState, FlowState, CopilotState {
  hardwareStats: HardwareStats;

  login: (user: AppUser) => void;
  logout: () => void;

  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  addToast: (message: string, type: Toast['type']) => void;
  removeToast: (id: string) => void;
  setSetupComplete: (v: boolean) => void;
  setNimApiKey: (key: string | null) => void;
  dismissShortcutHint: () => void;
  updateHardwareStats: (stats: Partial<HardwareStats>) => void;

  setDatasetQuery: (q: string) => void;
  setDatasetSource: (s: 'HuggingFace' | 'OpenImages') => void;
  setTargetSize: (n: number) => void;
  startProcessing: () => void;
  resumeProcessing: () => void;
  stopProcessing: () => void;
  setPaused: (paused: boolean, question?: PipelineQuestion | null) => void;
  clearPendingQuestion: () => void;
  addTerminalLog: (log: TerminalLog) => void;
  clearTerminalLogs: () => void;
  setBlurData: (data: DatasetState['blurData']) => void;
  setJobId: (id: string | null) => void;
  setDatasetResults: (results: DatasetResult[]) => void;
  setImagesIngested: (n: number | null) => void;
  setQualityScore: (n: number | null) => void;
  setClassBalance: (s: string | null) => void;
  setDuplicatePercentage: (n: number | null) => void;
  setProcessingPlan: (plan: ProcessingPlan | null) => void;
  setPreprocessingReport: (report: PreprocessingReport | null) => void;
  setClonedRecipe: (name: string | null) => void;

  startTraining: (runId: string, taskType: TrainingState['trainingTaskType']) => void;
  stopTraining: () => void;
  addTrainingLog: (log: TerminalLog) => void;
  clearTrainingLogs: () => void;
  appendTrainingMetric: (metric: TrainingMetricPoint) => void;
  setTrainingMetrics: (metrics: TrainingMetricPoint[]) => void;
  setTrainingTaskType: (taskType: TrainingState['trainingTaskType']) => void;

  addNode: (node: Node) => void;
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  setCopilotResponse: (r: string) => void;
  setIsCopilotStreaming: (v: boolean) => void;
}

const initialNodes: Node[] = [
  { id: 'input-1', type: 'inputNode', position: { x: 50, y: 200 }, data: { label: 'Image Input', channels: 3, resolution: '224×224' } },
  { id: 'conv-1', type: 'convBlock', position: { x: 350, y: 150 }, data: { label: 'Conv2d Block', filters: 64, kernel: '3×3', activation: 'ReLU' } },
  { id: 'bn-1', type: 'batchNormNode', position: { x: 650, y: 150 }, data: { label: 'BatchNorm2d', num_features: 64 } },
  { id: 'res-1', type: 'residualBlock', position: { x: 950, y: 150 }, data: { label: 'ResBlock', in_channels: 64, out_channels: 64, stride: 1 } },
  { id: 'attn-1', type: 'attentionBlock', position: { x: 1250, y: 200 }, data: { label: 'Self-Attention', heads: 8, dimK: 64 } },
  { id: 'drop-1', type: 'dropoutNode', position: { x: 1550, y: 200 }, data: { label: 'Dropout', p: 0.5 } },
  { id: 'linear-1', type: 'linearNode', position: { x: 1850, y: 200 }, data: { label: 'Linear', in_features: 512, out_features: 10, activation: 'ReLU' } },
  { id: 'out-1', type: 'outputNode', position: { x: 2150, y: 200 }, data: { label: 'Output Layer', num_classes: 10, activation: 'Softmax' } },
];

const initialEdges: Edge[] = [
  { id: 'e1', source: 'input-1', target: 'conv-1', animated: true },
  { id: 'e2', source: 'conv-1', target: 'bn-1', animated: true },
  { id: 'e3', source: 'bn-1', target: 'res-1', animated: true },
  { id: 'e4', source: 'res-1', target: 'attn-1', animated: true },
  { id: 'e5', source: 'attn-1', target: 'drop-1', animated: true },
  { id: 'e6', source: 'drop-1', target: 'linear-1', animated: true },
  { id: 'e7', source: 'linear-1', target: 'out-1', animated: true },
];

export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      isAuthenticated: false,
      user: null,
      login: (user) => set({ isAuthenticated: true, user }),
      logout: () => set({ isAuthenticated: false, user: null }),

      sidebarOpen: false,
      toasts: [],
      setupComplete: false,
      nimApiKey: null,
      shortcutHintDismissed: false,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      addToast: (message, type) =>
        set((s) => ({ toasts: [...s.toasts.slice(-2), { id: crypto.randomUUID(), message, type }] })),
      removeToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
      setSetupComplete: (v) => set({ setupComplete: v }),
      setNimApiKey: (key) => set({ nimApiKey: key }),
      dismissShortcutHint: () => set({ shortcutHintDismissed: true }),

      hardwareStats: { gpuModel: 'NVIDIA RTX 4090', vramUsed: 8.4, vramTotal: 24, cpuPercent: 34, ramPercent: 61 },
      updateHardwareStats: (stats) => set((s) => ({ hardwareStats: { ...s.hardwareStats, ...stats } })),

      isProcessing: false,
      isPaused: false,
      pendingQuestion: null,
      localDataset: null,
      terminalLogs: [],
      datasetQuery: '',
      datasetSource: 'HuggingFace',
      targetSize: 500,
      blurData: [],
      jobId: null,
      datasetResults: [],
      imagesIngested: null,
      qualityScore: null,
      classBalance: null,
      duplicatePercentage: null,
      processingPlan: null,
      preprocessingReport: null,
      clonedRecipe: null,

      setDatasetQuery: (q) => set({ datasetQuery: q }),
      setDatasetSource: (s) => set({ datasetSource: s }),
      setTargetSize: (n) => set({ targetSize: n }),
      startProcessing: () => set({ isProcessing: true, isPaused: false, pendingQuestion: null, terminalLogs: [] }),
      resumeProcessing: () => set({ isProcessing: true, isPaused: false, pendingQuestion: null }),
      stopProcessing: () => set({ isProcessing: false, isPaused: false }),
      setPaused: (paused, question = null) => set({ isPaused: paused, pendingQuestion: question }),
      setLocalDataset: (data) => set({ localDataset: data }),
      clearPendingQuestion: () => set({ pendingQuestion: null }),

      addTerminalLog: (log) =>
        set((state) => {
          if (state.terminalLogs.some((l) => l.id === log.id)) return state;
          return { terminalLogs: [...state.terminalLogs, log] };
        }),
      clearTerminalLogs: () => set({ terminalLogs: [], isProcessing: false, isPaused: false, jobId: null, pendingQuestion: null }),
      setBlurData: (data) => set({ blurData: data }),
      setJobId: (id) => set({ jobId: id }),
      setDatasetResults: (results) => set({ datasetResults: results }),
      setImagesIngested: (n) => set({ imagesIngested: n }),
      setQualityScore: (n) => set({ qualityScore: n }),
      setClassBalance: (s) => set({ classBalance: s }),
      setDuplicatePercentage: (n) => set({ duplicatePercentage: n }),
      setProcessingPlan: (plan) => set({ processingPlan: plan }),
      setPreprocessingReport: (report) => set({ preprocessingReport: report }),
      setClonedRecipe: (name) => set({ clonedRecipe: name }),

      isTraining: false,
      trainingRunId: null,
      trainingTaskType: 'mnist_classification',
      trainingLogs: [],
      trainingMetrics: [],
      startTraining: (runId, taskType) =>
        set({
          isTraining: true,
          trainingRunId: runId,
          trainingTaskType: taskType,
          trainingLogs: [],
          trainingMetrics: [],
        }),
      stopTraining: () => set({ isTraining: false }),
      addTrainingLog: (log) =>
        set((state) => {
          if (state.trainingLogs.some((l) => l.id === log.id)) return state;
          return { trainingLogs: [...state.trainingLogs, log] };
        }),
      clearTrainingLogs: () => set({ trainingLogs: [], trainingMetrics: [], trainingRunId: null, isTraining: false }),
      appendTrainingMetric: (metric) =>
        set((state) => {
          const existing = state.trainingMetrics.filter((item) => item.epoch !== metric.epoch || item.run_id !== metric.run_id);
          return { trainingMetrics: [...existing, metric].sort((a, b) => a.epoch - b.epoch) };
        }),
      setTrainingMetrics: (metrics) => set({ trainingMetrics: metrics }),
      setTrainingTaskType: (taskType) => set({ trainingTaskType: taskType }),

      nodes: initialNodes,
      edges: initialEdges,
      onNodesChange: (changes) => set({ nodes: applyNodeChanges(changes, get().nodes) }),
      onEdgesChange: (changes) => set({ edges: applyEdgeChanges(changes, get().edges) }),
      onConnect: (connection) => set({ edges: addEdge({ ...connection, animated: true }, get().edges) }),
      addNode: (node) => set((state) => ({ nodes: [...state.nodes, node] })),
      setNodes: (nodes) => set({ nodes }),
      setEdges: (edges) => set({ edges }),
      updateNodeData: (nodeId, data) => set((state) => ({
        nodes: state.nodes.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n))
      })),

      copilotResponse: '',
      isCopilotStreaming: false,
      setCopilotResponse: (r) => set({ copilotResponse: r }),
      setIsCopilotStreaming: (v) => set({ isCopilotStreaming: v }),
    }),
    {
      name: 'cvagent-auth',
      partialize: (state) => ({
        isAuthenticated: state.isAuthenticated,
        user: state.user,
        sidebarOpen: state.sidebarOpen,
        setupComplete: state.setupComplete,
        nimApiKey: state.nimApiKey,
        shortcutHintDismissed: state.shortcutHintDismissed,
      }),
    }
  )
);
