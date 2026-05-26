import { useMemo, useCallback, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  useReactFlow,
  ReactFlowProvider,
  type NodeTypes,
  type EdgeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useAppStore } from '../store/useAppStore';
import { computeGraphShapes } from '../lib/shapes';
import InputNode from '../components/builder/nodes/InputNode';
import ConvBlock from '../components/builder/nodes/ConvBlock';
import AttentionBlock from '../components/builder/nodes/AttentionBlock';
import ResidualBlock from '../components/builder/nodes/ResidualBlock';
import PoolingNode from '../components/builder/nodes/PoolingNode';
import LinearNode from '../components/builder/nodes/LinearNode';
import BatchNormNode from '../components/builder/nodes/BatchNormNode';
import DropoutNode from '../components/builder/nodes/DropoutNode';
import OutputNode from '../components/builder/nodes/OutputNode';
import ComponentPalette from '../components/builder/ComponentPalette';
import CopilotPanel from '../components/builder/CopilotPanel';
import TrainingTerminal from '../components/builder/TrainingTerminal';
import DeletableEdge from '../components/builder/DeletableEdge';

function BuilderCanvas() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode } = useAppStore();
  const { screenToFlowPosition, fitView } = useReactFlow();

  const shapedNodes = useMemo(() => computeGraphShapes(nodes, edges), [nodes, edges]);

  const nodeTypes: NodeTypes = useMemo(
    () => ({
      inputNode: InputNode,
      convBlock: ConvBlock,
      attentionBlock: AttentionBlock,
      residualBlock: ResidualBlock,
      poolingNode: PoolingNode,
      linearNode: LinearNode,
      batchNormNode: BatchNormNode,
      dropoutNode: DropoutNode,
      outputNode: OutputNode,
    }),
    []
  );

  const edgeTypes: EdgeTypes = useMemo(
    () => ({
      deletableEdge: DeletableEdge,
    }),
    []
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      const dataStr = event.dataTransfer.getData('application/reactflow-data');

      if (typeof type === 'undefined' || !type) {
        return;
      }

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: JSON.parse(dataStr || '{}'),
      };

      addNode(newNode);
    },
    [screenToFlowPosition, addNode]
  );

  useEffect(() => {
    if (nodes.length > 0) {
      const handle = window.setTimeout(() => {
        void fitView({ padding: 0.35, duration: 400 });
      }, 40);
      return () => window.clearTimeout(handle);
    }
  }, [fitView, nodes]);

  return (
    <div className="flex-1 relative h-full">
      <ReactFlow
        nodes={shapedNodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.4 }}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
          type: 'deletableEdge',
          animated: true,
          style: { stroke: '#475569', strokeWidth: 2 },
        }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#334155"
        />
        <Controls
          showInteractive={false}
          position="bottom-left"
        />
        <MiniMap
          nodeStrokeWidth={3}
          nodeColor={(node) => {
            if (node.type === 'inputNode') return '#10b981';
            if (node.type === 'convBlock') return '#38bdf8';
            if (node.type === 'attentionBlock') return '#a78bfa';
            if (node.type === 'residualBlock') return '#f59e0b';
            if (node.type === 'poolingNode') return '#fb7185';
            if (node.type === 'linearNode') return '#2dd4bf';
            if (node.type === 'batchNormNode') return '#94a3b8';
            if (node.type === 'dropoutNode') return '#fbbf24';
            if (node.type === 'outputNode') return '#059669';
            return '#64748b';
          }}
          maskColor="rgba(10, 10, 10, 0.7)"
          position="bottom-right"
        />
      </ReactFlow>

      {/* Canvas watermark */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 pointer-events-none">
        <span className="text-[10px] font-mono text-slate-700 tracking-widest uppercase">
          Visual Architecture Builder
        </span>
      </div>
    </div>
  );
}

export default function BuilderPage() {
  const { clonedRecipe, setClonedRecipe } = useAppStore();

  useEffect(() => {
    document.title = 'Builder — VisCurator';
    if (clonedRecipe) {
      const t = setTimeout(() => setClonedRecipe(null), 4000);
      return () => clearTimeout(t);
    }
  }, [clonedRecipe, setClonedRecipe]);

  return (
    <div className="flex h-full relative">
      {clonedRecipe && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 bg-sky-500/20 border border-sky-500/50 text-sky-100 px-6 py-2 rounded-xl shadow-lg backdrop-blur-md animate-fade-down flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" />
          <span className="text-sm font-semibold tracking-wide">Loaded: {clonedRecipe}</span>
        </div>
      )}
      <ReactFlowProvider>
        {/* Left: Component Palette */}
        <ComponentPalette />

        {/* Center: React Flow Canvas + Training Terminal */}
        <div className="flex-1 h-full flex flex-col min-w-0">
          <BuilderCanvas />
          <div className="px-4 pb-4">
            <TrainingTerminal />
          </div>
        </div>

        {/* Right: AI Co-Pilot */}
        <CopilotPanel />
      </ReactFlowProvider>
    </div>
  );
}
