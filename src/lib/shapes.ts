import { type Node, type Edge } from '@xyflow/react';

export function computeGraphShapes(nodes: Node[], edges: Edge[]): Node[] {
  // 1. Build adjacency list
  const incoming = new Map<string, string[]>();
  edges.forEach((e) => {
    if (!incoming.has(e.target)) incoming.set(e.target, []);
    incoming.get(e.target)!.push(e.source);
  });

  // 2. Find start nodes (inputNode)
  const inputNodes = nodes.filter((n) => n.type === 'inputNode');
  
  // 3. State to hold computed resolutions
  const computed = new Map<string, { res: number; channels: number }>();

  // Initialize inputs
  inputNodes.forEach((n) => {
    const resStr = n.data.resolution as string || '224x224';
    const match = resStr.match(/(\d+)/);
    const res = match ? parseInt(match[1], 10) : 224;
    const c = (n.data.channels as number) || 3;
    computed.set(n.id, { res, channels: c });
  });

  // 4. Topological sort / BFS
  const queue = [...inputNodes.map(n => n.id)];
  const visited = new Set<string>();

  while (queue.length > 0) {
    const currentId = queue.shift()!;
    if (visited.has(currentId)) continue;
    
    // Check if all inputs are computed (simple check)
    const inEdges = incoming.get(currentId) || [];
    if (!inEdges.every(id => computed.has(id))) {
      // Re-queue for later if not ready
      queue.push(currentId);
      // Prevent infinite loop if disconnected/cycle
      if (queue.length > nodes.length * 2) break;
      continue;
    }
    
    visited.add(currentId);

    const node = nodes.find(n => n.id === currentId);
    if (!node) continue;

    // Get input shape (just take first one for now)
    let inShape = inEdges.length > 0 ? computed.get(inEdges[0]) : computed.get(currentId);
    if (!inShape) inShape = { res: 224, channels: 3 };

    let outRes = inShape.res;
    let outChannels = inShape.channels;

    // Compute shape based on node type
    switch (node.type) {
      case 'convBlock':
        // Assume padding='same' for 3x3 so resolution stays same.
        // We could parse kernel if needed, but standard is 'same'
        outChannels = (node.data.filters as number) || outChannels;
        break;
      case 'poolingNode':
        // Standard 2x2 max pool with stride 2
        const pStride = (node.data.stride as number) || 2;
        if (node.data.pool_type === 'AdaptiveAvg') {
           outRes = 1;
        } else {
           outRes = Math.floor(outRes / pStride);
        }
        break;
      case 'residualBlock':
        const rStride = (node.data.stride as number) || 1;
        outRes = Math.floor(outRes / rStride);
        outChannels = (node.data.out_channels as number) || outChannels;
        break;
      case 'attentionBlock':
        // Resolution usually stays same
        break;
      case 'linearNode':
      case 'dropoutNode':
      case 'batchNormNode':
      case 'outputNode':
        // Resolution conceptually 1x1 or meaningless, keep as is or set to 1
        break;
    }

    computed.set(currentId, { res: outRes, channels: outChannels });

    // Queue neighbors
    edges.forEach(e => {
      if (e.source === currentId) queue.push(e.target);
    });
  }

  // 5. Apply computed shapes to nodes
  return nodes.map(n => {
    const shape = computed.get(n.id);
    if (shape && n.type !== 'inputNode') { // inputNode manages its own string
      return {
        ...n,
        data: {
          ...n.data,
          computed_res: `${shape.res}×${shape.res}`,
          computed_channels: shape.channels
        }
      };
    }
    return n;
  });
}
