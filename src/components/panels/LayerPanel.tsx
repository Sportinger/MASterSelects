// Slots panel component - dynamic grid with drag & drop

import { useState, useCallback, Fragment } from 'react';
import { Logger } from '../../services/logger';

const log = Logger.create('LayerPanel');
import { useMixerStore } from '../../stores/mixerStore';
import { openFilePicker } from '../../utils/fileLoader';

type DropZone = 'full' | 'left' | 'right';

export function LayerPanel() {
  const {
    layers,
    selectedLayerId,
    selectLayer,
    setLayerVisibility,
    clearLayerSource,
    swapSlots,
    triggerColumn,
    triggerRow,
    gridColumns,
    gridRows,
    setGridColumns,
    setGridRows,
    slotGroups,
    duplicateSlot,
    createSlotGroup,
    moveGroup,
  } = useMixerStore();

  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [draggedGroup, setDraggedGroup] = useState<number[] | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const [dropZone, setDropZone] = useState<DropZone>('full');
  const [isCtrlDrag, setIsCtrlDrag] = useState(false);
  const [highlightedColumn, setHighlightedColumn] = useState<number | null>(null);
  const [highlightedRow, setHighlightedRow] = useState<number | null>(null);
  const [hoveredSlot, setHoveredSlot] = useState<number | null>(null);
  const [hoveredGroup, setHoveredGroup] = useState<number[] | null>(null);

  const gridSize = gridColumns * gridRows;

  const handleColumnTrigger = useCallback((columnIndex: number) => {
    triggerColumn(columnIndex);
    setHighlightedColumn(columnIndex);
    setTimeout(() => setHighlightedColumn(null), 300);
  }, [triggerColumn]);

  const handleRowTrigger = useCallback((rowIndex: number) => {
    triggerRow(rowIndex);
    setHighlightedRow(rowIndex);
    setTimeout(() => setHighlightedRow(null), 300);
  }, [triggerRow]);

  const handleSlotTrigger = useCallback((slotIndex: number) => {
    // Empty slot clicked - deactivate all clips in this row
    const rowIndex = Math.floor(slotIndex / gridColumns);
    const { layers } = useMixerStore.getState();

    // Hide all clips in this row
    for (let col = 0; col < gridColumns; col++) {
      const idx = rowIndex * gridColumns + col;
      const layer = layers[idx];
      if (layer) {
        useMixerStore.getState().setLayerVisibility(layer.id, false);
      }
    }
  }, [gridColumns]);

  const handleAddMedia = async (slotIndex: number) => {
    const files = await openFilePicker({ accept: ['video/*', 'image/*'] });
    if (files.length > 0) {
      const state = useMixerStore.getState();
      state.createLayerAtSlot(slotIndex, files[0]);
    }
  };

  // Helper to get drop zone from mouse position
  const getDropZone = (e: React.DragEvent, hasContent: boolean): DropZone => {
    if (!hasContent) return 'full';
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const halfWidth = rect.width / 2;
    return x < halfWidth ? 'left' : 'right';
  };

  // Helper to find which group a slot belongs to
  const getSlotGroupIndex = (slotIndex: number): number => {
    return slotGroups.findIndex(group => group.includes(slotIndex));
  };

  // Drag handlers
  const handleDragStart = (e: React.DragEvent, index: number) => {
    setDraggedIndex(index);
    setIsCtrlDrag(e.ctrlKey);

    // Check if this slot is part of a group
    const groupIndex = getSlotGroupIndex(index);
    const isGroupDrag = groupIndex !== -1;

    if (isGroupDrag) {
      setDraggedGroup(slotGroups[groupIndex]);
    } else {
      setDraggedGroup(null);
    }

    e.dataTransfer.effectAllowed = e.ctrlKey ? 'copy' : 'move';
    e.dataTransfer.setData('text/plain', index.toString());
    e.dataTransfer.setData('ctrlKey', e.ctrlKey ? 'true' : 'false');
    e.dataTransfer.setData('isGroupDrag', isGroupDrag ? 'true' : 'false');
    if (isGroupDrag) {
      e.dataTransfer.setData('groupSlots', JSON.stringify(slotGroups[groupIndex]));
    }

    if (e.currentTarget instanceof HTMLElement) {
      e.currentTarget.classList.add('dragging');
    }
  };

  const handleDragEnd = (e: React.DragEvent) => {
    setDraggedIndex(null);
    setDraggedGroup(null);
    setDragOverIndex(null);
    setDropZone('full');
    setIsCtrlDrag(false);
    if (e.currentTarget instanceof HTMLElement) {
      e.currentTarget.classList.remove('dragging');
    }
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    const targetLayer = layers[index];
    const hasContent = !!targetLayer?.source;

    // Check if dragging files from outside
    if (e.dataTransfer.types.includes('Files')) {
      e.dataTransfer.dropEffect = 'copy';
      setDragOverIndex(index);
      setDropZone(getDropZone(e, hasContent));
    } else if (draggedIndex !== null && draggedIndex !== index) {
      // Update Ctrl state during drag
      setIsCtrlDrag(e.ctrlKey);
      e.dataTransfer.dropEffect = e.ctrlKey ? 'copy' : 'move';
      setDragOverIndex(index);
      setDropZone(getDropZone(e, hasContent));
    }
  };

  const handleDragLeave = () => {
    setDragOverIndex(null);
    setDropZone('full');
  };

  // Get adjacent slot index based on drop zone (left or right of current slot)
  const getAdjacentSlot = (index: number, zone: DropZone): number | null => {
    const col = index % gridColumns;
    if (zone === 'left') {
      // Left side -> slot to the left
      if (col === 0) return null; // Already at left edge
      return index - 1;
    } else if (zone === 'right') {
      // Right side -> slot to the right
      if (col === gridColumns - 1) return null; // Already at right edge
      return index + 1;
    }
    return null;
  };

  const handleDrop = (e: React.DragEvent, toIndex: number) => {
    e.preventDefault();
    const targetLayer = layers[toIndex];
    const hasContent = !!targetLayer?.source;
    const zone = getDropZone(e, hasContent);

    log.debug('Drop event', { toIndex, hasContent, zone });

    // Handle external file drop
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      log.debug('External file drop', { name: file.name, type: file.type });
      if (file.type.startsWith('video/') || file.type.startsWith('image/')) {
        const state = useMixerStore.getState();

        if (hasContent && zone !== 'full') {
          // Drop on half of filled slot - put in adjacent slot (left or right)
          const adjacentSlot = getAdjacentSlot(toIndex, zone);
          log.debug('Half-slot drop', { adjacentSlot });
          if (adjacentSlot !== null && !layers[adjacentSlot]?.source) {
            state.createLayerAtSlot(adjacentSlot, file);
            // Create a group between the existing slot and the new one
            const groupSlots = zone === 'left' ? [adjacentSlot, toIndex] : [toIndex, adjacentSlot];
            createSlotGroup(groupSlots);
          }
        } else {
          // Normal drop on empty slot or full replacement
          log.debug('Normal drop to slot', { toIndex });
          state.createLayerAtSlot(toIndex, file);
        }
      }
      setDragOverIndex(null);
      setDropZone('full');
      return;
    }

    // Handle internal slot operations
    const fromIndex = parseInt(e.dataTransfer.getData('text/plain'), 10);
    const wasCtrlDrag = e.ctrlKey || e.dataTransfer.getData('ctrlKey') === 'true';
    const isGroupDrag = e.dataTransfer.getData('isGroupDrag') === 'true';
    const groupSlotsData = e.dataTransfer.getData('groupSlots');

    log.debug('Internal drag', { fromIndex, wasCtrlDrag, isGroupDrag });

    if (!isNaN(fromIndex) && fromIndex !== toIndex) {
      if (wasCtrlDrag) {
        // Ctrl+drag = duplicate
        log.debug('Duplicating slot');
        duplicateSlot(fromIndex, toIndex);
      } else if (isGroupDrag && groupSlotsData) {
        // Moving a group - move all slots together
        const draggedGroupSlots = JSON.parse(groupSlotsData) as number[];
        log.debug('Moving group', { draggedGroupSlots, toIndex });
        moveGroup(draggedGroupSlots, toIndex);
      } else if (hasContent && zone !== 'full') {
        // Drag to half of filled slot - move to adjacent slot and group
        const adjacentSlot = getAdjacentSlot(toIndex, zone);
        log.debug('Half-slot internal', { adjacentSlot });
        if (adjacentSlot !== null && !layers[adjacentSlot]?.source) {
          swapSlots(fromIndex, adjacentSlot);
          // Create a group between the target slot and the moved slot
          const groupSlots = zone === 'left' ? [adjacentSlot, toIndex] : [toIndex, adjacentSlot];
          createSlotGroup(groupSlots);
        }
      } else {
        // Normal swap
        log.debug('Normal swap');
        swapSlots(fromIndex, toIndex);
      }
    }
    setDraggedIndex(null);
    setDragOverIndex(null);
    setDropZone('full');
    setIsCtrlDrag(false);
  };

  const handleSlotClick = (layer: typeof layers[0]) => {
    if (!layer) return;
    // Toggle selection: deselect if already selected, otherwise select
    if (selectedLayerId === layer.id) {
      selectLayer(null);
    } else {
      selectLayer(layer.id);
    }
  };

  // Check if a slot has a grouped neighbor on left or right
  const getGroupPosition = (index: number): { isLeft: boolean; isRight: boolean; isMiddle: boolean } => {
    const groupIndex = getSlotGroupIndex(index);
    if (groupIndex === -1) return { isLeft: false, isRight: false, isMiddle: false };

    const group = slotGroups[groupIndex];
    const col = index % gridColumns;

    // Check if left neighbor is in the same group
    const leftNeighbor = col > 0 ? index - 1 : null;
    const hasLeftGrouped = leftNeighbor !== null && group.includes(leftNeighbor);

    // Check if right neighbor is in the same group
    const rightNeighbor = col < gridColumns - 1 ? index + 1 : null;
    const hasRightGrouped = rightNeighbor !== null && group.includes(rightNeighbor);

    return {
      isLeft: !hasLeftGrouped && hasRightGrouped,  // First in group (left edge)
      isRight: hasLeftGrouped && !hasRightGrouped, // Last in group (right edge)
      isMiddle: hasLeftGrouped && hasRightGrouped, // Middle of group
    };
  };

  const renderSlot = (index: number) => {
    const layer = layers[index];
    const isHovered = hoveredSlot === index;
    const isSelected = layer && selectedLayerId === layer.id;
    const groupIndex = getSlotGroupIndex(index);
    const isInGroup = groupIndex !== -1;
    const currentGroup = isInGroup ? slotGroups[groupIndex] : null;
    const isGroupHovered = hoveredGroup !== null && hoveredGroup.includes(index);
    const isDragOverHalf = dragOverIndex === index && dropZone !== 'full' && layer?.source;
    const groupPos = getGroupPosition(index);
    const isBeingDraggedAsGroup = draggedGroup !== null && draggedGroup.includes(index);

    return (
      <div
        key={index}
        className={`slot-item ${layer ? 'has-layer' : 'empty'} ${isSelected ? 'selected' : ''} ${layer?.source ? 'has-media' : ''} ${draggedIndex === index || isBeingDraggedAsGroup ? 'dragging' : ''} ${dragOverIndex === index ? 'drag-over' : ''} ${highlightedColumn === index % gridColumns ? 'column-highlight' : ''} ${highlightedRow === Math.floor(index / gridColumns) ? 'row-highlight' : ''} ${isInGroup ? `in-group group-${groupIndex % 6}` : ''} ${groupPos.isLeft ? 'group-left' : ''} ${groupPos.isRight ? 'group-right' : ''} ${groupPos.isMiddle ? 'group-middle' : ''} ${isGroupHovered ? 'group-highlight' : ''} ${isDragOverHalf ? `drop-zone-${dropZone}` : ''} ${isCtrlDrag && dragOverIndex === index ? 'copy-mode' : ''}`}
        onClick={() => handleSlotClick(layer)}
        onMouseEnter={() => {
          setHoveredSlot(index);
          if (currentGroup) setHoveredGroup(currentGroup);
        }}
        onMouseLeave={() => {
          setHoveredSlot(null);
          setHoveredGroup(null);
        }}
        draggable={!!layer?.source}
        onDragStart={(e) => handleDragStart(e, index)}
        onDragEnd={handleDragEnd}
        onDragOver={(e) => handleDragOver(e, index)}
        onDragLeave={handleDragLeave}
        onDrop={(e) => handleDrop(e, index)}
      >
        {/* Half-slot drop zone indicators */}
        {isDragOverHalf && (
          <>
            <div className={`drop-zone-indicator left ${dropZone === 'left' ? 'active' : ''}`} />
            <div className={`drop-zone-indicator right ${dropZone === 'right' ? 'active' : ''}`} />
          </>
        )}
        {layer ? (
          <>
            <div className="slot-header">
              <button
                className={`visibility-btn ${layer.visible ? 'visible' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setLayerVisibility(layer.id, !layer.visible);
                }}
              >
                {layer.visible ? '●' : '○'}
              </button>
              <span className="slot-number">{index + 1}</span>
            </div>

            <div className="slot-preview">
              {layer.source?.videoElement ? (
                <video
                  ref={(el) => {
                    if (el) {
                      if (isHovered) {
                        el.play().catch(() => {});
                      } else {
                        el.pause();
                      }
                    }
                  }}
                  src={layer.source.videoElement.src}
                  muted
                  loop
                  playsInline
                  className="slot-thumbnail"
                />
              ) : layer.source?.imageElement ? (
                <img
                  src={layer.source.imageElement.src}
                  alt={layer.name}
                  className="slot-thumbnail"
                />
              ) : (
                <div
                  className="slot-empty-media"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleAddMedia(index);
                  }}
                >
                  <span>+</span>
                </div>
              )}
            </div>

            {layer.source && isSelected && (
              <div className="slot-controls">
                <button
                  className="btn-clear"
                  onClick={(e) => {
                    e.stopPropagation();
                    clearLayerSource(layer.id);
                  }}
                  title="Clear"
                >
                  ×
                </button>
              </div>
            )}
          </>
        ) : (
          <div
            className="slot-placeholder"
            onClick={() => handleSlotTrigger(index)}
          >
            <span
              className="slot-add"
              onClick={(e) => {
                e.stopPropagation();
                handleAddMedia(index);
              }}
            >
              +
            </span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="layer-panel slots-panel">
      <div className="panel-header">
        <h3>Slots</h3>
        <div className="grid-controls">
          <button
            className="grid-btn"
            onClick={() => setGridColumns(gridColumns - 1)}
            title="Remove column"
          >
            ←
          </button>
          <button
            className="grid-btn"
            onClick={() => setGridColumns(gridColumns + 1)}
            title="Add column"
          >
            →
          </button>
          <button
            className="grid-btn"
            onClick={() => setGridRows(gridRows - 1)}
            title="Remove row"
          >
            ↑
          </button>
          <button
            className="grid-btn"
            onClick={() => setGridRows(gridRows + 1)}
            title="Add row"
          >
            ↓
          </button>
        </div>
        <span className="slot-count">{layers.filter(l => l?.source).length} / {gridSize}</span>
      </div>

      <div
        className="slots-grid-with-triggers"
        style={{
          gridTemplateColumns: `28px repeat(${gridColumns}, 1fr)`,
        }}
      >
        {/* Empty corner */}
        <div className="trigger-corner" />

        {/* Column triggers */}
        {Array.from({ length: gridColumns }, (_, i) => (
          <button
            key={`col-${i}`}
            className={`column-trigger-btn ${highlightedColumn === i ? 'active' : ''}`}
            onClick={() => handleColumnTrigger(i)}
            title={`Trigger Column ${i + 1}`}
          >
            {i + 1}
          </button>
        ))}

        {/* Row triggers and slots */}
        {Array.from({ length: gridRows }, (_, rowIndex) => (
          <Fragment key={`row-${rowIndex}`}>
            <button
              className={`row-trigger-btn ${highlightedRow === rowIndex ? 'active' : ''}`}
              onClick={() => handleRowTrigger(rowIndex)}
              title={`Trigger Row ${rowIndex + 1}`}
            >
              {String.fromCharCode(65 + rowIndex)}
            </button>
            {Array.from({ length: gridColumns }, (_, colIndex) =>
              renderSlot(rowIndex * gridColumns + colIndex)
            )}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
