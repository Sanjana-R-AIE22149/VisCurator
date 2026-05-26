import { useEffect, useRef, useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '../../store/useAppStore';
import { Bot, Zap, AlertTriangle, ChevronRight, X, Copy, Check, Download, BarChart2, ArrowRight } from 'lucide-react';
import { streamCopilotAnalysis, compileGraph, type CompileResult } from '../../lib/copilot';
import { connectTrainingWebSocket, getHealth, startTrainingRun, getLocalDatasets, inspectDataset, type PipelineMessage, type DatasetInfo } from '../../lib/api';

function highlightPython(code: string): React.ReactNode[] {
  const lines = code.split('\n');
  return lines.map((line, i) => {
    const segments: React.ReactNode[] = [];
    let remaining = line;
    let key = 0;

    const patterns: Array<{ regex: RegExp; className: string }> = [
      { regex: /#.*$/, className: 'text-slate-500 italic' },
      { regex: /""".*?"""/, className: 'text-teal-400' },
      { regex: /"(?:[^"\\]|\\.)*"/, className: 'text-teal-400' },
      { regex: /'(?:[^'\\]|\\.)*'/, className: 'text-teal-400' },
      { regex: /f"(?:[^"\\]|\\.)*"/, className: 'text-teal-400' },
      {
        regex: /\b(import|from|class|def|return|self|if|else|elif|for|in|as|super|True|False|None|pass|and|or|not|with)\b/,
        className: 'text-violet-400 font-medium',
      },
      {
        regex: /\b(print|sum|len|range|int|float|str|list|dict|tuple|type)\b/,
        className: 'text-amber-400',
      },
      { regex: /\b\d+\.?\d*\b/, className: 'text-sky-400' },
      { regex: /\bnn\.\w+/, className: 'text-emerald-400' },
      { regex: /\btorch\.\w+/, className: 'text-emerald-400' },
    ];

    while (remaining.length > 0) {
      let earliest: { index: number; length: number; className: string } | null = null;

      for (const pattern of patterns) {
        const match = pattern.regex.exec(remaining);
        if (match && (earliest === null || match.index < earliest.index)) {
          earliest = { index: match.index, length: match[0].length, className: pattern.className };
        }
      }

      if (!earliest) {
        segments.push(<span key={key++}>{remaining}</span>);
        break;
      }

      if (earliest.index > 0) {
        segments.push(<span key={key++}>{remaining.slice(0, earliest.index)}</span>);
      }

      segments.push(
        <span key={key++} className={earliest.className}>
          {remaining.slice(earliest.index, earliest.index + earliest.length)}
        </span>
      );

      remaining = remaining.slice(earliest.index + earliest.length);
    }

    return (
      <div key={i} className="flex">
        <span className="mr-4 inline-block w-8 shrink-0 select-none text-right text-slate-600">
          {i + 1}
        </span>
        <span>{segments}</span>
      </div>
    );
  });
}

export default function CopilotPanel() {
  const navigate = useNavigate();
  const {
    copilotResponse,
    setCopilotResponse,
    isCopilotStreaming,
    setIsCopilotStreaming,
    nodes,
    edges,
    startTraining,
    stopTraining,
    addTrainingLog,
    appendTrainingMetric,
    trainingTaskType,
    setTrainingTaskType,
    isTraining,
    trainingRunId,
    preprocessingReport,
    updateNodeData,
  } = useAppStore();

  const scrollRef = useRef<HTMLDivElement>(null);
  const trainingWsRef = useRef<any>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastGraphRef = useRef('');

  const [showCompileModal, setShowCompileModal] = useState(false);
  const [compileResult, setCompileResult] = useState<CompileResult | null>(null);
  const [isCompiling, setIsCompiling] = useState(false);
  const [compileError, setCompileError] = useState('');
  const [copied, setCopied] = useState(false);
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  
  const [customEpochs, setCustomEpochs] = useState<number>(10);
  const [customImages, setCustomImages] = useState<number>(1000);
  
  const [localDatasets, setLocalDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetPath, setSelectedDatasetPath] = useState<string>('');

  useEffect(() => {
    getLocalDatasets()
      .then(ds => {
        setLocalDatasets(ds);
        if (ds.length > 0) {
          const recentPath = preprocessingReport?.output_dir || ds[ds.length - 1].path;
          setSelectedDatasetPath(recentPath);
          
          // Auto-sync if it's already set to custom_curated
          if (useAppStore.getState().trainingTaskType === 'custom_curated') {
            const match = ds.find(d => d.path === recentPath);
            if (match && match.metadata) {
              const inputNode = nodes.find(n => n.type === 'inputNode');
              if (inputNode) {
                const res = match.metadata.resolution ? `${match.metadata.resolution}×${match.metadata.resolution}` : '224×224';
                updateNodeData(inputNode.id, { resolution: res });
              }
              const outputNode = nodes.find(n => n.type === 'outputNode');
              if (outputNode && match.metadata.num_classes) {
                updateNodeData(outputNode.id, { num_classes: match.metadata.num_classes });
              }
            }
          }
        }
      })
      .catch(err => console.log('Failed to fetch datasets', err));
  }, [preprocessingReport, updateNodeData, nodes]); // nodes added for finding input/output on load

  // When a custom dataset is selected, inspect it to get real resolution & class count
  useEffect(() => {
    if (trainingTaskType !== 'custom_curated' || !selectedDatasetPath) return;
    inspectDataset(selectedDatasetPath)
      .then(info => {
        const inputNode = nodes.find(n => n.type === 'inputNode');
        if (inputNode) {
          const w = info.sample_width;
          const h = info.sample_height;
          updateNodeData(inputNode.id, { resolution: `${w}×${h}` });
        }
        const outputNode = nodes.find(n => n.type === 'outputNode');
        if (outputNode) {
          updateNodeData(outputNode.id, { num_classes: info.num_classes });
        }
      })
      .catch(() => {/* ignore, don't break UI */});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDatasetPath, trainingTaskType]);

  const handleDatasetChange = (path: string) => {
    setSelectedDatasetPath(path);
    const ds = localDatasets.find(d => d.path === path);
    if (ds && ds.metadata) {
      const inputNode = nodes.find(n => n.type === 'inputNode');
      if (inputNode) {
        const res = ds.metadata.resolution ? `${ds.metadata.resolution}×${ds.metadata.resolution}` : '224×224';
        updateNodeData(inputNode.id, { resolution: res });
      }
      const outputNode = nodes.find(n => n.type === 'outputNode');
      if (outputNode && ds.metadata.num_classes) {
        updateNodeData(outputNode.id, { num_classes: ds.metadata.num_classes });
      }
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [copilotResponse]);

  const analyzeGraph = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setCopilotResponse('');
    setIsCopilotStreaming(true);

    const alive = await getHealth();
    if (!alive.online) {
      if (!signal.aborted) {
        setCopilotResponse('Backend offline.\n\nStart `python run_demo.py` to enable real copilot analysis.');
        setIsCopilotStreaming(false);
      }
      return;
    }

    try {
      let accumulated = '';
      for await (const chunk of streamCopilotAnalysis(nodes, edges, signal)) {
        if (signal.aborted) return;
        accumulated += chunk;
        setCopilotResponse(accumulated);
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return;
      if (!signal.aborted) {
        setCopilotResponse(`Analysis unavailable.\n\n${err instanceof Error ? err.message : 'Streaming request failed.'}`);
        setIsCopilotStreaming(false);
      }
      return;
    }

    if (!signal.aborted) {
      setIsCopilotStreaming(false);
    }
  }, [edges, nodes, setCopilotResponse, setIsCopilotStreaming]);

  useEffect(() => {
    const signature = JSON.stringify({
      n: nodes.map(n => ({ t: n.type, d: n.data })),
      e: edges.map(e => ({ s: e.source, t: e.target }))
    });
    if (signature === lastGraphRef.current) return;
    lastGraphRef.current = signature;

    const timer = setTimeout(analyzeGraph, 1200);
    return () => clearTimeout(timer);
  }, [nodes, edges, analyzeGraph]);

  useEffect(() => {
    return () => {
      trainingWsRef.current?.close();
    };
  }, []);

  const handleCompile = useCallback(async () => {
    setIsCompiling(true);
    setCompileResult(null);
    setCompileError('');
    setShowCompileModal(true);

    try {
      const result = await compileGraph(nodes, edges);
      setCompileResult(result);
    } catch (err) {
      setCompileError(err instanceof Error ? err.message : 'Unknown compilation error');
    }

    setIsCompiling(false);
  }, [edges, nodes]);

  const handleCopy = async () => {
    if (!compileResult?.code) return;
    try {
      await navigator.clipboard.writeText(compileResult.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available
    }
  };

  const handleDownload = () => {
    if (!compileResult?.code) return;
    const blob = new Blob([compileResult.code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'cvagent_model.py';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const mapStatus = (type: PipelineMessage['type']): 'ok' | 'warn' | 'error' | 'info' => {
    switch (type) {
      case 'done':
        return 'ok';
      case 'error':
        return 'error';
      case 'tool_result':
        return 'ok';
      default:
        return 'info';
    }
  };

  const formatTrainingMessage = (msg: PipelineMessage): string => {
    if (msg.type === 'tool_result' && typeof msg.data?.epoch === 'number') {
      const data = msg.data as Record<string, unknown>;
      return `epoch ${data.epoch} | loss ${Number(data.loss ?? 0).toFixed(4)} | acc ${Number(data.accuracy ?? 0).toFixed(3)} | mAP ${Number(data.map ?? 0).toFixed(3)} | ETA ${String(data.eta ?? '--:--')}`;
    }
    switch (msg.type) {
      case 'done':
        return `complete | ${msg.message}`;
      case 'error':
        return `error | ${msg.message}`;
      default:
        return msg.message;
    }
  };

  const handleTrainingMessage = (msg: PipelineMessage) => {
    addTrainingLog({
      id: msg.id || `train-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
      message: formatTrainingMessage(msg),
      status: mapStatus(msg.type),
      msgType: msg.type,
    });

    if (msg.type === 'tool_result' && typeof msg.data?.epoch === 'number') {
      appendTrainingMetric(msg.data as any);
    }

    if (msg.type === 'done' || msg.type === 'error') {
      stopTraining();
      trainingWsRef.current?.close();
    }
  };

  const handleTrain = useCallback(async () => {
    setIsStartingTraining(true);
    const datasetPath = trainingTaskType === 'custom_curated' ? selectedDatasetPath : null;
    try {
      const response = await startTrainingRun(
        nodes, 
        edges, 
        trainingTaskType, 
        datasetPath,
        trainingTaskType === 'custom_curated' ? customEpochs : undefined,
        trainingTaskType === 'custom_curated' ? customImages : undefined
      );
      startTraining(response.run_id, response.task_type);
      addTrainingLog({
        id: `train-start-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `queued run ${response.run_id} | ${response.task_type}${datasetPath ? ` | dataset: ${datasetPath}` : ' | synthetic data'}`,
        status: 'info',
        msgType: 'log',
      });
      trainingWsRef.current?.close();
      trainingWsRef.current = connectTrainingWebSocket(
        response.run_id,
        handleTrainingMessage,
        (event) => {
          if (useAppStore.getState().isTraining) {
            addTrainingLog({
              id: `train-close-${Date.now()}`,
              timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
              message: event && event.code !== 1000 ? `Training socket closed (${event.code}). Reconnecting...` : 'Training socket closed.',
              status: 'info',
              msgType: 'log',
            });
          }
        },
        () => {
          addTrainingLog({
            id: `train-ws-error-${Date.now()}`,
            timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
            message: 'Training websocket error. Reconnecting...',
            status: 'warn',
            msgType: 'error',
          });
        }
      );
    } catch (err) {
      addTrainingLog({
        id: `train-error-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: err instanceof Error ? err.message : 'Failed to start training',
        status: 'error',
        msgType: 'error',
      });
      stopTraining();
    } finally {
      setIsStartingTraining(false);
    }
  }, [addTrainingLog, edges, nodes, selectedDatasetPath, startTraining, stopTraining, trainingTaskType, customEpochs, customImages]);

  return (
    <>
      {/* (Rest of the component stays the same) */}
      <div className="h-full w-80 animate-slide-in-right border-l border-slate-800/60 bg-slate-950/80 backdrop-blur-xl flex flex-col">
        <div className="flex items-center gap-2 border-b border-slate-800/50 px-4 py-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg border border-violet-500/25 bg-violet-500/15">
            <Bot className="h-3.5 w-3.5 text-violet-400" />
          </div>
          <div>
            <h3 className="text-xs font-semibold text-slate-200">AI Co-Pilot</h3>
            <p className="font-mono text-[9px] text-slate-500">
              {isCopilotStreaming ? 'Analyzing...' : 'Ready'}
            </p>
          </div>
          {isCopilotStreaming && <div className="ml-auto h-1.5 w-1.5 rounded-full bg-violet-400 animate-pulse" />}
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 font-mono text-[11px] leading-relaxed text-slate-400">
          {!copilotResponse ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-slate-600">
              <Bot className="h-8 w-8 opacity-30" />
              <p className="text-[10px]">Awaiting analysis...</p>
            </div>
          ) : (
            <div className="space-y-0.5 whitespace-pre-wrap">
              {copilotResponse.split('\n').map((line, i) => {
                if (line.startsWith('Warning:') || line.startsWith('warning:')) {
                  return (
                    <div key={i} className="flex items-start gap-1.5 text-amber-400/90">
                      <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
                      <span>{line.replace(/^warning:\s*/i, '').replace(/^Warning:\s*/, '')}</span>
                    </div>
                  );
                }
                if (line.startsWith('**')) {
                  return (
                    <p key={i} className="font-medium text-slate-300">
                      {line.replace(/\*\*/g, '')}
                    </p>
                  );
                }
                if (line.startsWith('```')) {
                  return <span key={i} />;
                }
                if (line.startsWith('  ')) {
                  return (
                    <p key={i} className="rounded bg-slate-900/50 px-2 py-0.5 text-teal-400/80">
                      {line}
                    </p>
                  );
                }
                return <p key={i}>{line}</p>;
              })}
              {isCopilotStreaming && <span className="terminal-cursor" />}
            </div>
          )}
        </div>

        <div className="space-y-2 border-t border-slate-800/50 p-4">
          <button
            onClick={analyzeGraph}
            disabled={isCopilotStreaming}
            className="w-full flex items-center justify-center gap-2 rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-xs text-slate-400 transition-all duration-200 hover:bg-slate-800 hover:text-slate-300 disabled:opacity-40"
          >
            <Zap className="h-3 w-3" />
            Re-analyze Graph
          </button>
          <button
            onClick={handleCompile}
            disabled={isCompiling}
            className="btn-glow w-full flex items-center justify-center gap-2 rounded-lg border border-teal-500/40 bg-gradient-to-r from-teal-500/20 to-emerald-500/20 px-3 py-2.5 text-xs font-semibold tracking-wide text-teal-300 shadow-[0_0_20px_rgba(45,212,191,0.15)] transition-all duration-300 hover:from-teal-500/30 hover:to-emerald-500/30 hover:border-teal-500/60 disabled:opacity-40"
          >
            <ChevronRight className="h-3.5 w-3.5" />
            {isCompiling ? 'Compiling...' : 'Compile to PyTorch'}
          </button>
          <div className="space-y-2 rounded-lg border border-slate-800/60 bg-slate-900/40 p-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[10px] uppercase tracking-widest text-slate-500">Training Task</span>
              {trainingRunId && <span className="font-mono text-[9px] text-slate-600">{trainingRunId}</span>}
            </div>
            <select
              value={trainingTaskType}
              onChange={(e) => setTrainingTaskType(e.target.value as any)}
              disabled={isTraining || isStartingTraining}
              className="w-full rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2 text-[11px] text-slate-200 focus:outline-none focus:ring-1 focus:ring-teal-500/50"
            >
              <option value="mnist_classification">MNIST CNN Classification</option>
              <option value="object_detection">Light Object Detection</option>
              <option value="custom_curated">Custom Curated Dataset</option>
            </select>
            {trainingTaskType === 'custom_curated' && (
              <div className="space-y-2">
                <div>
                  <span className="text-[9px] uppercase tracking-widest text-slate-500">Dataset</span>
                  <select
                    value={selectedDatasetPath}
                    onChange={(e) => handleDatasetChange(e.target.value)}
                    className="w-full mt-1 rounded bg-slate-900 border border-slate-700 px-2 py-1.5 text-[11px] text-slate-200"
                  >
                    {localDatasets.length === 0 && <option value="">No datasets found</option>}
                    {localDatasets.map(ds => (
                      <option key={ds.id} value={ds.path}>{ds.name}</option>
                    ))}
                  </select>
                </div>
                <div className="flex gap-2">
                  <div className="flex-1">
                    <span className="text-[9px] uppercase tracking-widest text-slate-500">Epochs</span>
                    <input type="number" min={1} value={customEpochs} onChange={(e) => setCustomEpochs(Number(e.target.value))} className="w-full mt-1 rounded bg-slate-900 border border-slate-700 px-2 py-1.5 text-[11px] text-slate-200" />
                  </div>
                  <div className="flex-1">
                    <span className="text-[9px] uppercase tracking-widest text-slate-500">Subset Size</span>
                    <input type="number" min={1} value={customImages} onChange={(e) => setCustomImages(Number(e.target.value))} className="w-full mt-1 rounded bg-slate-900 border border-slate-700 px-2 py-1.5 text-[11px] text-slate-200" />
                  </div>
                </div>
              </div>
            )}
            <button
              onClick={handleTrain}
              disabled={isTraining || isStartingTraining}
              className="w-full flex items-center justify-center gap-2 rounded-lg border border-sky-500/35 bg-sky-500/15 px-3 py-2.5 text-xs font-semibold text-sky-300 transition-all duration-200 hover:bg-sky-500/25 hover:border-sky-500/50 disabled:opacity-40"
            >
              <ChevronRight className="h-3.5 w-3.5" />
              {isTraining || isStartingTraining ? 'Training...' : 'Train Model'}
            </button>
          </div>
          <p className="pt-1 text-center font-mono text-[8px] tracking-wider text-slate-600">
            Powered by NVIDIA NIM
          </p>

          {/* ── View Analytics CTA (after training run completes) ── */}
          {trainingRunId && !isTraining && (
            <button
              id="cta-view-analytics"
              onClick={() => navigate(`/analytics?runId=${trainingRunId}`)}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-gradient-to-r from-teal-600/20 to-emerald-600/20 border border-teal-500/30 text-teal-300 text-xs font-semibold transition-all hover:from-teal-600/30 hover:border-teal-500/50 shadow-lg shadow-teal-500/10"
            >
              <BarChart2 className="h-3.5 w-3.5" />
              View Training Analytics
              <ArrowRight className="h-3 w-3 ml-auto" />
            </button>
          )}
        </div>
      </div>

      {showCompileModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="flex max-h-[90vh] w-[90vw] max-w-[900px] flex-col overflow-hidden rounded-xl border border-slate-700/60 bg-slate-950 shadow-2xl">
            <div className="shrink-0 flex items-center justify-between border-b border-slate-800/60 px-5 py-3">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${isCompiling ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'}`} />
                  <span className="text-xs font-semibold text-slate-200">
                    {isCompiling ? 'Generating...' : 'Generated PyTorch Code'}
                  </span>
                </div>
                {compileResult && (
                  <span className="font-mono text-[10px] text-slate-500">
                    {compileResult.model_summary.split('\n')[1] ?? 'Model summary'}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={handleCopy}
                  disabled={!compileResult?.code}
                  className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[10px] text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200 disabled:opacity-30"
                >
                  {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                  {copied ? 'Copied!' : 'Copy Code'}
                </button>
                <button
                  onClick={handleDownload}
                  disabled={!compileResult?.code}
                  className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[10px] text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200 disabled:opacity-30"
                >
                  <Download className="h-3 w-3" />
                  Download .py
                </button>
                <div className="mx-1 h-4 w-px bg-slate-800" />
                <button
                  onClick={() => setShowCompileModal(false)}
                  className="rounded-lg p-1.5 text-slate-500 transition-colors hover:bg-slate-800 hover:text-slate-200"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            {compileResult && compileResult.warnings.length > 0 && (
              <div className="shrink-0 border-b border-slate-800/60 bg-amber-500/5 px-5 py-2.5">
                <div className="mb-1.5 flex items-center gap-2">
                  <AlertTriangle className="h-3 w-3 text-amber-400" />
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-amber-400">
                    Warnings ({compileResult.warnings.length})
                  </span>
                </div>
                <ul className="space-y-0.5">
                  {compileResult.warnings.map((warning, i) => (
                    <li key={i} className="pl-5 font-mono text-[11px] text-amber-400/80">
                      - {warning}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex-1 overflow-y-auto bg-[#0a0a0f] p-5">
              {isCompiling && (
                <div className="flex items-center gap-2 font-mono text-xs text-slate-500">
                  <div className="h-1.5 w-1.5 rounded-full bg-teal-400 animate-pulse" />
                  Generating PyTorch code...
                </div>
              )}
              {compileError && (
                <div className="font-mono text-xs text-red-400">
                  <p className="mb-1 font-semibold">Compilation Failed</p>
                  <p>{compileError}</p>
                </div>
              )}
              {compileResult && (
                <pre className="font-mono text-[11px] leading-[1.7]">
                  {highlightPython(compileResult.code)}
                </pre>
              )}
            </div>

            <div className="shrink-0 flex items-center justify-between border-t border-slate-800/60 px-5 py-2.5">
              <span className="font-mono text-[8px] tracking-wider text-slate-600">
                VisCurator Code Generator - Deterministic Compilation
              </span>
              <button
                onClick={() => setShowCompileModal(false)}
                className="rounded-lg border border-slate-700/50 bg-slate-800/60 px-3 py-1.5 text-xs text-slate-400 transition-colors hover:text-slate-200"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
