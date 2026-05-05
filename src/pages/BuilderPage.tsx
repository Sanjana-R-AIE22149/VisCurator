import { useMemo, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  useReactFlow,
  ReactFlowProvider,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useAppStore } from '../store/useAppStore';
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

function BuilderCanvas() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode } = useAppStore();
  const { screenToFlowPosition } = useReactFlow();

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

  return (
    <div className="flex-1 relative h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.4 }}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
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
  return (
    <div className="flex h-full">
      <ReactFlowProvider>
        {/* Left: Component Palette */}
        <ComponentPalette />

        {/* Center: React Flow Canvas */}
        <BuilderCanvas />

        {/* Right: AI Co-Pilot */}
        <CopilotPanel />
      </ReactFlowProvider>
    </div>
  );
}
