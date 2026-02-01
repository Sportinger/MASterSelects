// Dock layout tree manipulation utilities

import type {
  DockLayout,
  DockNode,
  DockPanel,
  DockTabGroup,
  DockSplit,
  DropTarget,
} from '../types/dock';

// Generate unique ID
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Find a node by ID in the tree
export function findNodeById(root: DockNode, id: string): DockNode | null {
  if (root.id === id) return root;
  if (root.kind === 'split') {
    const left = findNodeById(root.children[0], id);
    if (left) return left;
    return findNodeById(root.children[1], id);
  }
  return null;
}

// Find parent of a node
export function findParentOfNode(
  root: DockNode,
  nodeId: string
): { parent: DockSplit; childIndex: 0 | 1 } | null {
  if (root.kind === 'split') {
    if (root.children[0].id === nodeId) {
      return { parent: root, childIndex: 0 };
    }
    if (root.children[1].id === nodeId) {
      return { parent: root, childIndex: 1 };
    }
    const left = findParentOfNode(root.children[0], nodeId);
    if (left) return left;
    return findParentOfNode(root.children[1], nodeId);
  }
  return null;
}

// Remove a panel from a group
export function removePanel(
  layout: DockLayout,
  panelId: string,
  groupId: string
): DockLayout {
  return {
    ...layout,
    root: removePanelFromNode(layout.root, panelId, groupId),
  };
}

function removePanelFromNode(node: DockNode, panelId: string, groupId: string): DockNode {
  if (node.kind === 'tab-group') {
    if (node.id === groupId) {
      const newPanels = node.panels.filter((p) => p.id !== panelId);
      const newActiveIndex = Math.min(node.activeIndex, Math.max(0, newPanels.length - 1));
      return { ...node, panels: newPanels, activeIndex: newActiveIndex };
    }
    return node;
  }
  return {
    ...node,
    children: [
      removePanelFromNode(node.children[0], panelId, groupId),
      removePanelFromNode(node.children[1], panelId, groupId),
    ] as [DockNode, DockNode],
  };
}

// Insert a panel at a drop target
export function insertPanelAtTarget(
  layout: DockLayout,
  panel: DockPanel,
  target: DropTarget
): DockLayout {
  return {
    ...layout,
    root: insertPanelInNode(layout.root, panel, target),
  };
}

function insertPanelInNode(node: DockNode, panel: DockPanel, target: DropTarget): DockNode {
  if (node.id === target.groupId) {
    if (node.kind === 'tab-group') {
      if (target.position === 'center') {
        // Add as new tab at specified index (or end if not specified)
        const insertIndex = target.tabInsertIndex ?? node.panels.length;
        const newPanels = [...node.panels];
        newPanels.splice(insertIndex, 0, panel);
        return {
          ...node,
          panels: newPanels,
          activeIndex: insertIndex,
        };
      }
      // Split the group
      const direction = target.position === 'left' || target.position === 'right' ? 'horizontal' : 'vertical';
      const newGroup: DockTabGroup = {
        kind: 'tab-group',
        id: `group-${generateId()}`,
        panels: [panel],
        activeIndex: 0,
      };
      const isFirst = target.position === 'left' || target.position === 'top';
      return {
        kind: 'split',
        id: `split-${generateId()}`,
        direction,
        ratio: 0.5,
        children: isFirst ? [newGroup, node] : [node, newGroup],
      };
    }
  }
  if (node.kind === 'split') {
    return {
      ...node,
      children: [
        insertPanelInNode(node.children[0], panel, target),
        insertPanelInNode(node.children[1], panel, target),
      ] as [DockNode, DockNode],
    };
  }
  return node;
}

// Collapse single-child splits and empty groups
export function collapseSingleChildSplits(node: DockNode): DockNode {
  if (node.kind === 'tab-group') {
    // Empty groups become a placeholder (should be handled by caller)
    return node;
  }

  // Recursively collapse children first
  const child0 = collapseSingleChildSplits(node.children[0]);
  const child1 = collapseSingleChildSplits(node.children[1]);

  // Check if either child is an empty tab group
  const child0Empty = child0.kind === 'tab-group' && child0.panels.length === 0;
  const child1Empty = child1.kind === 'tab-group' && child1.panels.length === 0;

  if (child0Empty && child1Empty) {
    // Both empty - return an empty group
    return {
      kind: 'tab-group',
      id: node.id,
      panels: [],
      activeIndex: 0,
    };
  }

  if (child0Empty) {
    return child1;
  }

  if (child1Empty) {
    return child0;
  }

  return {
    ...node,
    children: [child0, child1] as [DockNode, DockNode],
  };
}

// Get all panel IDs in the layout
export function getAllPanelIds(layout: DockLayout): string[] {
  const ids: string[] = [];
  collectPanelIds(layout.root, ids);
  layout.floatingPanels.forEach((f) => ids.push(f.panel.id));
  return ids;
}

function collectPanelIds(node: DockNode, ids: string[]): void {
  if (node.kind === 'tab-group') {
    node.panels.forEach((p) => ids.push(p.id));
  } else {
    collectPanelIds(node.children[0], ids);
    collectPanelIds(node.children[1], ids);
  }
}

// Calculate drop position from mouse coordinates
export function calculateDropPosition(
  rect: DOMRect,
  mouseX: number,
  mouseY: number
): 'center' | 'left' | 'right' | 'top' | 'bottom' {
  const relX = (mouseX - rect.left) / rect.width;
  const relY = (mouseY - rect.top) / rect.height;

  const edgeThreshold = 0.25;

  // Check edges first
  if (relX < edgeThreshold) return 'left';
  if (relX > 1 - edgeThreshold) return 'right';
  if (relY < edgeThreshold) return 'top';
  if (relY > 1 - edgeThreshold) return 'bottom';

  return 'center';
}

// Get all tab groups in the layout (for hit testing)
export function getAllTabGroups(node: DockNode): DockTabGroup[] {
  if (node.kind === 'tab-group') {
    return [node];
  }
  return [
    ...getAllTabGroups(node.children[0]),
    ...getAllTabGroups(node.children[1]),
  ];
}
