import type { Edge, Node } from '@xyflow/react';

export type RecipeTemplateId =
  | 'yolov8-nano'
  | 'resnet50'
  | 'efficientdet-d0'
  | 'detr';

export interface GraphTemplate {
  nodes: Node[];
  edges: Edge[];
}

const baseInput = (id: string, x: number, y: number, resolution = '640x640'): Node => ({
  id,
  type: 'inputNode',
  position: { x, y },
  data: { label: 'Image Input', channels: 3, resolution },
});

const baseOutput = (id: string, x: number, y: number, numClasses: number, activation: string): Node => ({
  id,
  type: 'outputNode',
  position: { x, y },
  data: { label: 'Output Layer', num_classes: numClasses, activation },
});

const templates: Record<RecipeTemplateId, GraphTemplate> = {
  'yolov8-nano': {
    nodes: [
      baseInput('input-1', 60, 220, '640x640'),
      { id: 'conv-1', type: 'convBlock', position: { x: 320, y: 120 }, data: { label: 'Conv2d Block', filters: 32, kernel: '3x3', activation: 'SiLU' } },
      { id: 'conv-2', type: 'convBlock', position: { x: 600, y: 120 }, data: { label: 'Conv2d Block', filters: 64, kernel: '3x3', activation: 'SiLU' } },
      { id: 'res-1', type: 'residualBlock', position: { x: 900, y: 120 }, data: { label: 'ResBlock', in_channels: 64, out_channels: 64, stride: 1, filters: 64 } },
      { id: 'conv-3', type: 'convBlock', position: { x: 1200, y: 120 }, data: { label: 'Conv2d Block', filters: 128, kernel: '3x3', activation: 'SiLU' } },
      { id: 'pool-1', type: 'poolingNode', position: { x: 1500, y: 120 }, data: { label: 'MaxPool2d', pool_type: 'Max', kernel: '2x2', stride: 2 } },
      { id: 'linear-1', type: 'linearNode', position: { x: 1780, y: 120 }, data: { label: 'Linear', in_features: 512, out_features: 256, bias: true, activation: 'SiLU' } },
      baseOutput('output-1', 2060, 120, 80, 'Sigmoid'),
    ],
    edges: [
      { id: 'e1', source: 'input-1', target: 'conv-1', animated: true },
      { id: 'e2', source: 'conv-1', target: 'conv-2', animated: true },
      { id: 'e3', source: 'conv-2', target: 'res-1', animated: true },
      { id: 'e4', source: 'res-1', target: 'conv-3', animated: true },
      { id: 'e5', source: 'conv-3', target: 'pool-1', animated: true },
      { id: 'e6', source: 'pool-1', target: 'linear-1', animated: true },
      { id: 'e7', source: 'linear-1', target: 'output-1', animated: true },
    ],
  },
  resnet50: {
    nodes: [
      baseInput('input-1', 60, 240, '224x224'),
      { id: 'conv-1', type: 'convBlock', position: { x: 320, y: 140 }, data: { label: 'Conv2d Block', filters: 64, kernel: '7x7', activation: 'ReLU' } },
      { id: 'bn-1', type: 'batchNormNode', position: { x: 620, y: 140 }, data: { label: 'BatchNorm2d', num_features: 64, eps: 1e-5, momentum: 0.1 } },
      { id: 'pool-1', type: 'poolingNode', position: { x: 900, y: 140 }, data: { label: 'MaxPool2d', pool_type: 'Max', kernel: '3x3', stride: 2 } },
      { id: 'res-1', type: 'residualBlock', position: { x: 1200, y: 140 }, data: { label: 'ResBlock', in_channels: 64, out_channels: 256, stride: 1, filters: 256 } },
      { id: 'res-2', type: 'residualBlock', position: { x: 1500, y: 140 }, data: { label: 'ResBlock', in_channels: 256, out_channels: 512, stride: 2, filters: 512 } },
      { id: 'res-3', type: 'residualBlock', position: { x: 1800, y: 140 }, data: { label: 'ResBlock', in_channels: 512, out_channels: 1024, stride: 2, filters: 1024 } },
      { id: 'linear-1', type: 'linearNode', position: { x: 2100, y: 140 }, data: { label: 'Linear', in_features: 2048, out_features: 1000, bias: true, activation: 'ReLU' } },
      baseOutput('output-1', 2380, 140, 1000, 'Softmax'),
    ],
    edges: [
      { id: 'e1', source: 'input-1', target: 'conv-1', animated: true },
      { id: 'e2', source: 'conv-1', target: 'bn-1', animated: true },
      { id: 'e3', source: 'bn-1', target: 'pool-1', animated: true },
      { id: 'e4', source: 'pool-1', target: 'res-1', animated: true },
      { id: 'e5', source: 'res-1', target: 'res-2', animated: true },
      { id: 'e6', source: 'res-2', target: 'res-3', animated: true },
      { id: 'e7', source: 'res-3', target: 'linear-1', animated: true },
      { id: 'e8', source: 'linear-1', target: 'output-1', animated: true },
    ],
  },
  'efficientdet-d0': {
    nodes: [
      baseInput('input-1', 60, 220, '512x512'),
      { id: 'conv-1', type: 'convBlock', position: { x: 320, y: 120 }, data: { label: 'Conv2d Block', filters: 32, kernel: '3x3', activation: 'SiLU' } },
      { id: 'res-1', type: 'residualBlock', position: { x: 620, y: 120 }, data: { label: 'ResBlock', in_channels: 32, out_channels: 64, stride: 1, filters: 64 } },
      { id: 'conv-2', type: 'convBlock', position: { x: 920, y: 120 }, data: { label: 'Conv2d Block', filters: 128, kernel: '3x3', activation: 'SiLU' } },
      { id: 'attention-1', type: 'attentionBlock', position: { x: 1220, y: 120 }, data: { label: 'Self-Attention', heads: 4, dimK: 32 } },
      { id: 'res-2', type: 'residualBlock', position: { x: 1520, y: 120 }, data: { label: 'ResBlock', in_channels: 128, out_channels: 128, stride: 1, filters: 128 } },
      { id: 'linear-1', type: 'linearNode', position: { x: 1820, y: 120 }, data: { label: 'Linear', in_features: 512, out_features: 256, bias: true, activation: 'SiLU' } },
      baseOutput('output-1', 2100, 120, 90, 'Sigmoid'),
    ],
    edges: [
      { id: 'e1', source: 'input-1', target: 'conv-1', animated: true },
      { id: 'e2', source: 'conv-1', target: 'res-1', animated: true },
      { id: 'e3', source: 'res-1', target: 'conv-2', animated: true },
      { id: 'e4', source: 'conv-2', target: 'attention-1', animated: true },
      { id: 'e5', source: 'attention-1', target: 'res-2', animated: true },
      { id: 'e6', source: 'res-2', target: 'linear-1', animated: true },
      { id: 'e7', source: 'linear-1', target: 'output-1', animated: true },
    ],
  },
  detr: {
    nodes: [
      baseInput('input-1', 60, 220, '800x800'),
      { id: 'conv-1', type: 'convBlock', position: { x: 320, y: 120 }, data: { label: 'Conv2d Block', filters: 64, kernel: '7x7', activation: 'ReLU' } },
      { id: 'res-1', type: 'residualBlock', position: { x: 620, y: 120 }, data: { label: 'ResBlock', in_channels: 64, out_channels: 256, stride: 1, filters: 256 } },
      { id: 'res-2', type: 'residualBlock', position: { x: 920, y: 120 }, data: { label: 'ResBlock', in_channels: 256, out_channels: 512, stride: 2, filters: 512 } },
      { id: 'attention-1', type: 'attentionBlock', position: { x: 1220, y: 120 }, data: { label: 'Self-Attention', heads: 8, dimK: 64 } },
      { id: 'dropout-1', type: 'dropoutNode', position: { x: 1520, y: 120 }, data: { label: 'Dropout', p: 0.1 } },
      { id: 'linear-1', type: 'linearNode', position: { x: 1820, y: 120 }, data: { label: 'Linear', in_features: 256, out_features: 256, bias: true, activation: 'ReLU' } },
      baseOutput('output-1', 2100, 120, 92, 'Sigmoid'),
    ],
    edges: [
      { id: 'e1', source: 'input-1', target: 'conv-1', animated: true },
      { id: 'e2', source: 'conv-1', target: 'res-1', animated: true },
      { id: 'e3', source: 'res-1', target: 'res-2', animated: true },
      { id: 'e4', source: 'res-2', target: 'attention-1', animated: true },
      { id: 'e5', source: 'attention-1', target: 'dropout-1', animated: true },
      { id: 'e6', source: 'dropout-1', target: 'linear-1', animated: true },
      { id: 'e7', source: 'linear-1', target: 'output-1', animated: true },
    ],
  },
};

export function getRecipeGraphTemplate(recipeId: RecipeTemplateId): GraphTemplate {
  const template = templates[recipeId];
  return {
    nodes: template.nodes.map((node) => ({
      ...node,
      position: { ...node.position },
      data: { ...(node.data as Record<string, unknown>) },
    })),
    edges: template.edges.map((edge) => ({ ...edge })),
  };
}
