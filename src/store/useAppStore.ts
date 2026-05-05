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

/* ── Auth ── */
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

/* ── UI ── */
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

/* ── Hardware Stats ── */
export interface HardwareStats {
  gpuModel: string;
  vramUsed: number;
  vramTotal: number;
  cpuPercent: number;
  ramPercent: number;
}

/* ── Terminal Log Entry ── */
export interface TerminalLog {
  id: string;
  timestamp: string;
  message: string;
  status: 'ok' | 'warn' | 'error' | 'info';
  msgType?: 'thought' | 'tool_call' | 'tool_result' | 'script_log' | 'log' | 'done' | 'error';
}

/* ── Dataset Result from Agent ── */
export interface DatasetResult {
  dataset_id: string;
  description: string;
  quality_score?: number;
  source: string;
  metadata?: Record<string, unknown>;
}

/* ── Dataset Pipeline State ── */
interface DatasetState {
  isProcessing: boolean;
  terminalLogs: TerminalLog[];
  datasetQuery: string;
  datasetSource: 'HuggingFace' | 'OpenImages';
  targetSize: number;
  blurData: Array<{ id: number; laplacian: number; resolution: number; accepted: boolean }>;
  jobId: string | null;
  datasetResults: DatasetResult[];
}

/* ── React Flow State ── */
interface FlowState {
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
}

/* ── AI Co-Pilot State ── */
interface CopilotState {
  copilotResponse: string;
  isCopilotStreaming: boolean;
}

/* ── Combined Store ── */
interface AppStore extends AuthState, UIState, DatasetState, FlowState, CopilotState {
  hardwareStats: HardwareStats;

  // Auth actions
  login: (user: AppUser) => void;
  logout: () => void;

  // UI actions
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  addToast: (message: string, type: Toast['type']) => void;
  removeToast: (id: string) => void;
  setSetupComplete: (v: boolean) => void;
  setNimApiKey: (key: string | null) => void;
  dismissShortcutHint: () => void;

  // Hardware actions
  updateHardwareStats: (stats: Partial<HardwareStats>) => void;

  // Dataset actions
  setDatasetQuery: (q: string) => void;
  setDatasetSource: (s: 'HuggingFace' | 'OpenImages') => void;
  setTargetSize: (n: number) => void;
  startProcessing: () => void;
  stopProcessing: () => void;
  addTerminalLog: (log: TerminalLog) => void;
  clearTerminalLogs: () => void;
  setBlurData: (data: DatasetState['blurData']) => void;
  setJobId: (id: string | null) => void;
  setDatasetResults: (results: DatasetResult[]) => void;

  // Flow actions
  addNode: (node: Node) => void;
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;

  // Copilot actions
  setCopilotResponse: (r: string) => void;
  setIsCopilotStreaming: (v: boolean) => void;
}

/* ── Initial Nodes ── */
const initialNodes: Node[] = [
  {
    id: 'input-1',
    type: 'inputNode',
    position: { x: 50, y: 200 },
    data: { label: 'Image Input', channels: 3, resolution: '224×224' },
  },
  {
    id: 'conv-1',
    type: 'convBlock',
    position: { x: 350, y: 150 },
    data: { label: 'Conv2d Block', filters: 64, kernel: '3×3', activation: 'ReLU' },
  },
  {
    id: 'bn-1',
    type: 'batchNormNode',
    position: { x: 650, y: 150 },
    data: { label: 'BatchNorm2d', num_features: 64, eps: 1e-5, momentum: 0.1 },
  },
  {
    id: 'res-1',
    type: 'residualBlock',
    position: { x: 950, y: 150 },
    data: { label: 'ResBlock', in_channels: 64, out_channels: 64, stride: 1 },
  },
  {
    id: 'attn-1',
    type: 'attentionBlock',
    position: { x: 1250, y: 200 },
    data: { label: 'Self-Attention', heads: 8, dimK: 64 },
  },
  {
    id: 'drop-1',
    type: 'dropoutNode',
    position: { x: 1550, y: 200 },
    data: { label: 'Dropout', p: 0.5 },
  },
  {
    id: 'linear-1',
    type: 'linearNode',
    position: { x: 1850, y: 200 },
    data: { label: 'Linear', in_features: 512, out_features: 10, bias: true, activation: 'ReLU' },
  },
  {
    id: 'out-1',
    type: 'outputNode',
    position: { x: 2150, y: 200 },
    data: { label: 'Output Layer', num_classes: 10, activation: 'Softmax' },
  },
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
      /* ── Auth ── */
      isAuthenticated: false,
      user: null,

      login: (user) => set({ isAuthenticated: true, user }),
      logout: () => set({ isAuthenticated: false, user: null }),

      /* ── UI ── */
      sidebarOpen: false,
      toasts: [],
      setupComplete: false,
      nimApiKey: null,
      shortcutHintDismissed: false,

      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      addToast: (message, type) =>
        set((s) => ({
          toasts: [
            ...s.toasts.slice(-2),
            { id: crypto.randomUUID(), message, type },
          ],
        })),
      removeToast: (id) =>
        set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
      setSetupComplete: (v) => set({ setupComplete: v }),
      setNimApiKey: (key) => set({ nimApiKey: key }),
      dismissShortcutHint: () => set({ shortcutHintDismissed: true }),

      /* ── Hardware ── */
      hardwareStats: {
        gpuModel: 'NVIDIA RTX 4090',
        vramUsed: 8.4,
        vramTotal: 24,
        cpuPercent: 34,
        ramPercent: 61,
      },
      updateHardwareStats: (stats) =>
        set((s) => ({ hardwareStats: { ...s.hardwareStats, ...stats } })),

      /* ── Dataset ── */
      isProcessing: false,
      terminalLogs: [],
      datasetQuery: '',
      datasetSource: 'HuggingFace',
      targetSize: 500,
      blurData: [],
      jobId: null,
      datasetResults: [],

      setDatasetQuery: (q) => set({ datasetQuery: q }),
      setDatasetSource: (s) => set({ datasetSource: s }),
      setTargetSize: (n) => set({ targetSize: n }),
      startProcessing: () => set({ isProcessing: true, terminalLogs: [] }),
      stopProcessing: () => set({ isProcessing: false }),
      addTerminalLog: (log) =>
        set((state) => ({ terminalLogs: [...state.terminalLogs, log] })),
      clearTerminalLogs: () => set({ terminalLogs: [], isProcessing: false, jobId: null }),
      setBlurData: (data) => set({ blurData: data }),
      setJobId: (id) => set({ jobId: id }),
      setDatasetResults: (results) => set({ datasetResults: results }),

      /* ── React Flow ── */
      nodes: initialNodes,
      edges: initialEdges,

      onNodesChange: (changes) =>
        set({ nodes: applyNodeChanges(changes, get().nodes) }),
      onEdgesChange: (changes) =>
        set({ edges: applyEdgeChanges(changes, get().edges) }),
      onConnect: (connection) =>
        set({ edges: addEdge({ ...connection, animated: true }, get().edges) }),

      addNode: (node) => set((state) => ({ nodes: [...state.nodes, node] })),
      setNodes: (nodes) => set({ nodes }),
      setEdges: (edges) => set({ edges }),

      /* ── Copilot ── */
      copilotResponse: '',
      isCopilotStreaming: false,

      setCopilotResponse: (r) => set({ copilotResponse: r }),
      setIsCopilotStreaming: (v) => set({ isCopilotStreaming: v }),
    }),
    {
      name: 'cvagent-auth',
      // Persist auth & critical UI state
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
