import { useEffect, useRef, useCallback, useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import { Bot, Zap, AlertTriangle, ChevronRight, X, Copy, Check, Download } from 'lucide-react';
import { streamCopilotAnalysis, compileGraph, type CompileResult } from '../../lib/copilot';
import { getHealth } from '../../lib/api';

/* ── Simulated AI analysis response (fallback when backend is offline) ── */
const AI_RESPONSE_CHUNKS = [
  'Analyzing current architecture graph...\n\n',
  '```\n',
  'Architecture Summary:\n',
  '  Input:  ImageTensor [3, 224, 224]\n',
  '  └─ Conv2d Block (64 filters, 3×3)\n',
  '      └─ Self-Attention (8 heads, d_k=64)\n',
  '```\n\n',
  '**Parameter Estimate:** 4.2M trainable parameters\n\n',
  '⚠ **Warning:** Potential bottleneck detected at spatial ',
  'dimension collapse between Conv2d and Self-Attention. ',
  'Consider adding adaptive pooling or a stride-2 conv ',
  'to gradually reduce spatial dimensions.\n\n',
  '**Recommendation:** Insert a `nn.AdaptiveAvgPool2d((14, 14))` ',
  'before the attention block to reduce computational cost by ~60%.\n\n',
  '**Estimated FLOPs:** 1.8 GFLOPs\n',
  '**Memory Footprint:** ~420 MB (batch_size=32)\n\n',
  '**Compatibility:** ✓ ONNX exportable\n',
  '**Quantization:** ✓ INT8 compatible (PTQ)\n',
];

/* ── Simple Python syntax highlighter ── */
function highlightPython(code: string): React.ReactNode[] {
  const lines = code.split('\n');
  return lines.map((line, i) => {
    const segments: React.ReactNode[] = [];
    let remaining = line;
    let key = 0;

    // Process each line for syntax highlighting
    const patterns: Array<{ regex: RegExp; className: string }> = [
      // Comments (must be first to avoid partial matching)
      { regex: /#.*$/, className: 'text-slate-500 italic' },
      // Triple-quoted strings
      { regex: /""".*?"""/, className: 'text-teal-400' },
      // Double-quoted strings
      { regex: /"(?:[^"\\]|\\.)*"/, className: 'text-teal-400' },
      // Single-quoted strings
      { regex: /'(?:[^'\\]|\\.)*'/, className: 'text-teal-400' },
      // f-strings
      { regex: /f"(?:[^"\\]|\\.)*"/, className: 'text-teal-400' },
      // Keywords
      {
        regex: /\b(import|from|class|def|return|self|if|else|elif|for|in|as|super|True|False|None|pass|and|or|not|with)\b/,
        className: 'text-violet-400 font-medium',
      },
      // Built-in functions & types
      {
        regex: /\b(print|sum|len|range|int|float|str|list|dict|tuple|type)\b/,
        className: 'text-amber-400',
      },
      // Numbers
      { regex: /\b\d+\.?\d*\b/, className: 'text-sky-400' },
      // nn.Module names
      {
        regex: /\bnn\.\w+/,
        className: 'text-emerald-400',
      },
      // torch references
      {
        regex: /\btorch\.\w+/,
        className: 'text-emerald-400',
      },
    ];

    // Simple sequential highlighting: find earliest match, emit text before + match, repeat
    while (remaining.length > 0) {
      let earliest: { index: number; length: number; className: string } | null = null;

      for (const pat of patterns) {
        const match = pat.regex.exec(remaining);
        if (match && (earliest === null || match.index < earliest.index)) {
          earliest = { index: match.index, length: match[0].length, className: pat.className };
        }
      }

      if (!earliest) {
        segments.push(<span key={key++}>{remaining}</span>);
        break;
      }

      // Text before the match
      if (earliest.index > 0) {
        segments.push(<span key={key++}>{remaining.slice(0, earliest.index)}</span>);
      }
      // The match itself
      segments.push(
        <span key={key++} className={earliest.className}>
          {remaining.slice(earliest.index, earliest.index + earliest.length)}
        </span>
      );
      remaining = remaining.slice(earliest.index + earliest.length);
    }

    return (
      <div key={i} className="flex">
        <span className="inline-block w-8 text-right mr-4 text-slate-600 select-none shrink-0">
          {i + 1}
        </span>
        <span>{segments}</span>
      </div>
    );
  });
}

export default function CopilotPanel() {
  const {
    copilotResponse,
    setCopilotResponse,
    isCopilotStreaming,
    setIsCopilotStreaming,
    nodes,
    edges,
  } = useAppStore();

  const scrollRef = useRef<HTMLDivElement>(null);

  // Compile modal state
  const [showCompileModal, setShowCompileModal] = useState(false);
  const [compileResult, setCompileResult] = useState<CompileResult | null>(null);
  const [isCompiling, setIsCompiling] = useState(false);
  const [compileError, setCompileError] = useState('');
  const [copied, setCopied] = useState(false);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [copilotResponse]);

  /* ── Mock fallback ── */
  const runMockAnalysis = useCallback(() => {
    setCopilotResponse('');
    setIsCopilotStreaming(true);

    let cumulativeDelay = 0;
    AI_RESPONSE_CHUNKS.forEach((chunk, idx) => {
      cumulativeDelay += 60 + Math.random() * 100;
      setTimeout(() => {
        setCopilotResponse(
          AI_RESPONSE_CHUNKS.slice(0, idx + 1).join('')
        );
        if (idx === AI_RESPONSE_CHUNKS.length - 1) {
          setIsCopilotStreaming(false);
        }
      }, cumulativeDelay);
    });
  }, [setCopilotResponse, setIsCopilotStreaming]);

  /* ── Real streaming analysis ── */
  const analyzeGraph = useCallback(async () => {
    setCopilotResponse('');
    setIsCopilotStreaming(true);

    const alive = await getHealth();
    if (!alive) {
      console.warn('[CopilotPanel] Backend unreachable — falling back to mock.');
      runMockAnalysis();
      return;
    }

    try {
      let accumulated = '';
      for await (const chunk of streamCopilotAnalysis(nodes, edges)) {
        accumulated += chunk;
        setCopilotResponse(accumulated);
      }
    } catch (err) {
      console.warn('[CopilotPanel] Streaming failed, using mock:', err);
      runMockAnalysis();
      return;
    }

    setIsCopilotStreaming(false);
  }, [nodes, edges, setCopilotResponse, setIsCopilotStreaming, runMockAnalysis]);

  /* Auto-analyze on mount */
  useEffect(() => {
    const timer = setTimeout(analyzeGraph, 800);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── Compile to PyTorch ── */
  const handleCompile = useCallback(async () => {
    setIsCompiling(true);
    setCompileResult(null);
    setCompileError('');
    setShowCompileModal(true);

    try {
      const result = await compileGraph(nodes, edges);
      setCompileResult(result);
    } catch (err) {
      setCompileError(
        err instanceof Error ? err.message : 'Unknown compilation error'
      );
    }

    setIsCompiling(false);
  }, [nodes, edges]);

  /* ── Copy code to clipboard ── */
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

  /* ── Download as .py file ── */
  const handleDownload = () => {
    if (!compileResult?.code) return;
    const blob = new Blob([compileResult.code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cvagent_model.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div className="w-80 h-full border-l border-slate-800/60 bg-slate-950/80 backdrop-blur-xl flex flex-col animate-slide-in-right">
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-800/50">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-violet-500/15 border border-violet-500/25">
            <Bot className="w-3.5 h-3.5 text-violet-400" />
          </div>
          <div>
            <h3 className="text-xs font-semibold text-slate-200">AI Co-Pilot</h3>
            <p className="text-[9px] text-slate-500 font-mono">
              {isCopilotStreaming ? 'Analyzing...' : 'Ready'}
            </p>
          </div>
          {isCopilotStreaming && (
            <div className="ml-auto w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
          )}
        </div>

        {/* Response area */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-4 py-3 font-mono text-[11px] leading-relaxed text-slate-400"
        >
          {!copilotResponse ? (
            <div className="flex flex-col items-center justify-center h-full gap-2 text-slate-600">
              <Bot className="w-8 h-8 opacity-30" />
              <p className="text-[10px]">Awaiting analysis...</p>
            </div>
          ) : (
            <div className="space-y-0.5 whitespace-pre-wrap">
              {copilotResponse.split('\n').map((line, i) => {
                if (line.startsWith('⚠')) {
                  return (
                    <div key={i} className="flex items-start gap-1.5 text-amber-400/90">
                      <AlertTriangle className="w-3 h-3 mt-0.5 shrink-0" />
                      <span>{line.replace('⚠ ', '')}</span>
                    </div>
                  );
                }
                if (line.startsWith('**')) {
                  return (
                    <p key={i} className="text-slate-300 font-medium">
                      {line.replace(/\*\*/g, '')}
                    </p>
                  );
                }
                if (line.startsWith('```')) {
                  return <span key={i} />;
                }
                if (line.startsWith('  ')) {
                  return (
                    <p key={i} className="text-teal-400/80 bg-slate-900/50 px-2 py-0.5 rounded">
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

        {/* Actions */}
        <div className="p-4 border-t border-slate-800/50 space-y-2">
          <button
            onClick={analyzeGraph}
            disabled={isCopilotStreaming}
            className="
              w-full flex items-center justify-center gap-2
              px-3 py-2 rounded-lg
              bg-slate-800/50 border border-slate-700/50
              text-xs text-slate-400
              hover:bg-slate-800 hover:text-slate-300
              disabled:opacity-40
              transition-all duration-200
            "
          >
            <Zap className="w-3 h-3" />
            Re-analyze Graph
          </button>
          <button
            onClick={handleCompile}
            disabled={isCompiling}
            className="
              w-full flex items-center justify-center gap-2
              px-3 py-2.5 rounded-lg
              bg-gradient-to-r from-teal-500/20 to-emerald-500/20
              border border-teal-500/40
              text-xs font-semibold text-teal-300 tracking-wide
              hover:from-teal-500/30 hover:to-emerald-500/30
              hover:border-teal-500/60
              disabled:opacity-40
              transition-all duration-300
              btn-glow
              shadow-[0_0_20px_rgba(45,212,191,0.15)]
            "
          >
            <ChevronRight className="w-3.5 h-3.5" />
            {isCompiling ? 'Compiling...' : 'Compile to PyTorch'}
          </button>

          {/* Powered by badge */}
          <p className="text-center text-[8px] text-slate-600 font-mono tracking-wider pt-1">
            Powered by NVIDIA NIM
          </p>
        </div>
      </div>

      {/* ── Full-Screen Compile Modal ── */}
      {showCompileModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="w-[90vw] max-w-[900px] max-h-[90vh] flex flex-col rounded-xl border border-slate-700/60 bg-slate-950 shadow-2xl overflow-hidden">
            {/* Modal header */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800/60 shrink-0">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isCompiling ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'}`} />
                  <span className="text-xs font-semibold text-slate-200">
                    {isCompiling ? 'Generating...' : 'Generated PyTorch Code'}
                  </span>
                </div>
                {compileResult && (
                  <span className="text-[10px] font-mono text-slate-500">
                    {compileResult.model_summary.split('\n')[1]}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={handleCopy}
                  disabled={!compileResult?.code}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[10px] text-slate-400 hover:text-slate-200 hover:bg-slate-800 disabled:opacity-30 transition-colors"
                >
                  {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  {copied ? 'Copied!' : 'Copy Code'}
                </button>
                <button
                  onClick={handleDownload}
                  disabled={!compileResult?.code}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[10px] text-slate-400 hover:text-slate-200 hover:bg-slate-800 disabled:opacity-30 transition-colors"
                >
                  <Download className="w-3 h-3" />
                  Download .py
                </button>
                <div className="w-px h-4 bg-slate-800 mx-1" />
                <button
                  onClick={() => setShowCompileModal(false)}
                  className="p-1.5 rounded-lg text-slate-500 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Warnings section */}
            {compileResult && compileResult.warnings.length > 0 && (
              <div className="px-5 py-2.5 border-b border-slate-800/60 bg-amber-500/5 shrink-0">
                <div className="flex items-center gap-2 mb-1.5">
                  <AlertTriangle className="w-3 h-3 text-amber-400" />
                  <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">
                    Warnings ({compileResult.warnings.length})
                  </span>
                </div>
                <ul className="space-y-0.5">
                  {compileResult.warnings.map((w, i) => (
                    <li key={i} className="text-[11px] font-mono text-amber-400/80 pl-5">
                      • {w}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Code body */}
            <div className="flex-1 overflow-y-auto p-5 bg-[#0a0a0f]">
              {isCompiling && (
                <div className="flex items-center gap-2 text-slate-500 font-mono text-xs">
                  <div className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
                  Generating PyTorch code...
                </div>
              )}
              {compileError && (
                <div className="text-red-400 font-mono text-xs">
                  <p className="font-semibold mb-1">Compilation Failed</p>
                  <p>{compileError}</p>
                </div>
              )}
              {compileResult && (
                <pre className="font-mono text-[11px] leading-[1.7]">
                  {highlightPython(compileResult.code)}
                </pre>
              )}
            </div>

            {/* Modal footer */}
            <div className="px-5 py-2.5 border-t border-slate-800/60 flex items-center justify-between shrink-0">
              <span className="text-[8px] text-slate-600 font-mono tracking-wider">
                VisCurator Code Generator • Deterministic Compilation
              </span>
              <button
                onClick={() => setShowCompileModal(false)}
                className="px-3 py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/50 text-xs text-slate-400 hover:text-slate-200 transition-colors"
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
