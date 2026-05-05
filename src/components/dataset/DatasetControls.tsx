import { useRef, useEffect } from 'react';
import { Search, ChevronDown, Play, RotateCcw } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { startDatasetPipeline, connectPipelineWebSocket, getHealth, type PipelineMessage } from '../../lib/api';

export default function DatasetControls() {
  const {
    datasetQuery,
    setDatasetQuery,
    datasetSource,
    setDatasetSource,
    targetSize,
    setTargetSize,
    isProcessing,
    startProcessing,
    stopProcessing,
    clearTerminalLogs,
    addTerminalLog,
    setJobId,
    setDatasetResults,
  } = useAppStore();

  const wsRef = useRef<WebSocket | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleFocus = () => {
      inputRef.current?.focus();
    };
    window.addEventListener('focus-dataset-query', handleFocus);
    return () => window.removeEventListener('focus-dataset-query', handleFocus);
  }, []);

  /* ── Map WS message type → terminal log status ── */
  const mapStatus = (type: PipelineMessage['type']): 'ok' | 'warn' | 'error' | 'info' => {
    switch (type) {
      case 'done':        return 'ok';
      case 'error':       return 'error';
      case 'tool_call':   return 'info';
      case 'tool_result': return 'ok';
      case 'thought':     return 'info';
      case 'script_log':  return 'info';
      default:            return 'info';
    }
  };

  /* ── Format the message text with prefix per type ── */
  const formatMessage = (msg: PipelineMessage): string => {
    switch (msg.type) {
      case 'thought':     return `💭 ${msg.message}`;
      case 'tool_call':   return `⚡ ${msg.message}`;
      case 'tool_result': return `✓ ${msg.message}`;
      case 'done':        return `✅ ${msg.message}`;
      case 'error':       return `✖ ${msg.message}`;
      case 'script_log':  return msg.message; // raw subprocess output — no prefix
      default:            return `> ${msg.message}`;
    }
  };

  /* ── Handle incoming WS messages ── */
  const handleWsMessage = (msg: PipelineMessage) => {
    addTerminalLog({
      id: `ws-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
      message: formatMessage(msg),
      status: mapStatus(msg.type),
      msgType: msg.type,
    });

    if (msg.type === 'done') {
      stopProcessing();
      // Extract dataset results from the final message data if available
      if (msg.data && typeof msg.data === 'object') {
        const toolResults = (msg.data as Record<string, unknown>).tool_results;
        if (Array.isArray(toolResults)) {
          setDatasetResults(toolResults);
        }
      }
    }

    if (msg.type === 'error') {
      stopProcessing();
    }
  };

  /* ── Execute button handler ── */
  const handleRun = async () => {
    if (!datasetQuery.trim()) return;
    startProcessing();

    // Check backend health first
    const backendAlive = await getHealth();

    if (!backendAlive) {
      console.warn('[DatasetControls] Backend unreachable — falling back to mock simulation.');
      addTerminalLog({
        id: `fallback-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: '> Backend offline — running mock simulation...',
        status: 'warn',
        msgType: 'log',
      });
      // The LiveTerminal mock pipeline will still run via the isProcessing flag
      return;
    }

    try {
      // 1. Create the pipeline job
      addTerminalLog({
        id: `init-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `> Connecting to CVAgent backend...`,
        status: 'info',
        msgType: 'log',
      });

      const { job_id } = await startDatasetPipeline(datasetQuery, datasetSource, targetSize);
      setJobId(job_id);

      addTerminalLog({
        id: `job-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `> Pipeline job created: ${job_id.slice(0, 8)}...`,
        status: 'ok',
        msgType: 'log',
      });

      // 2. Open WebSocket for real-time streaming
      wsRef.current = connectPipelineWebSocket(
        job_id,
        handleWsMessage,
        () => {
          // onClose — ensure processing state is cleaned up
          if (useAppStore.getState().isProcessing) {
            stopProcessing();
          }
        }
      );
    } catch (err) {
      console.error('[DatasetControls] Pipeline start failed:', err);
      addTerminalLog({
        id: `err-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `✖ Failed to start pipeline: ${err instanceof Error ? err.message : 'Unknown error'}`,
        status: 'error',
        msgType: 'error',
      });
      stopProcessing();
    }
  };

  const handleReset = () => {
    // Close any active WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    clearTerminalLogs();
    setJobId(null);
    setDatasetResults([]);
    useAppStore.setState({ blurData: [] });
  };

  return (
    <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 p-5 backdrop-blur-sm">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 items-end">
        {/* Dataset Query */}
        <div className="lg:col-span-1">
          <label className="block text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">
            Dataset Query
          </label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
            <input
              type="text"
              value={datasetQuery}
              onChange={(e) => setDatasetQuery(e.target.value)}
              placeholder="e.g. Apple Leaf Blight"
              disabled={isProcessing}
              className="
                w-full pl-9 pr-3 py-2.5
                bg-slate-950/80 border border-slate-800
                rounded-lg text-xs text-slate-200
                placeholder:text-slate-600
                focus:outline-none focus:ring-1 focus:ring-teal-500/50 focus:border-teal-500/40
                disabled:opacity-50
                transition-all duration-200
                font-mono
              "
            />
          </div>
        </div>

        {/* Source Dropdown */}
        <div>
          <label className="block text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">
            Source
          </label>
          <div className="relative">
            <select
              value={datasetSource}
              onChange={(e) => setDatasetSource(e.target.value as 'HuggingFace' | 'OpenImages')}
              disabled={isProcessing}
              className="
                w-full px-3 py-2.5
                bg-slate-950/80 border border-slate-800
                rounded-lg text-xs text-slate-200
                appearance-none cursor-pointer
                focus:outline-none focus:ring-1 focus:ring-teal-500/50 focus:border-teal-500/40
                disabled:opacity-50
                transition-all duration-200
              "
            >
              <option value="HuggingFace">HuggingFace Datasets</option>
              <option value="OpenImages">Google OpenImages</option>
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
          </div>
        </div>

        {/* Target Size Slider */}
        <div>
          <label className="block text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">
            Target Size: <span className="text-teal-400">{targetSize.toLocaleString()}</span>
          </label>
          <input
            type="range"
            min={100}
            max={5000}
            step={100}
            value={targetSize}
            onChange={(e) => setTargetSize(Number(e.target.value))}
            disabled={isProcessing}
            className="
              w-full h-1.5 bg-slate-800 rounded-full
              appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:w-3.5
              [&::-webkit-slider-thumb]:h-3.5
              [&::-webkit-slider-thumb]:bg-teal-400
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(45,212,191,0.5)]
              [&::-webkit-slider-thumb]:cursor-pointer
              disabled:opacity-50
            "
          />
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={handleRun}
            disabled={isProcessing || !datasetQuery.trim()}
            className="
              flex-1 flex items-center justify-center gap-2
              px-4 py-2.5 rounded-lg
              bg-teal-500/15 border border-teal-500/30
              text-teal-400 text-xs font-semibold tracking-wide
              hover:bg-teal-500/25 hover:border-teal-500/50
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-all duration-200
              btn-glow
            "
          >
            <Play className="w-3.5 h-3.5" />
            {isProcessing ? 'Running...' : 'Execute'}
          </button>
          <button
            onClick={handleReset}
            disabled={isProcessing}
            className="
              px-3 py-2.5 rounded-lg
              bg-slate-800/50 border border-slate-700/50
              text-slate-400 text-xs
              hover:bg-slate-800 hover:text-slate-300
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-all duration-200
            "
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
