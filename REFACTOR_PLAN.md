# Refactor Plan: Code Cleanup

## Status: In Progress

---

## Phase 1: Dead Code Entfernen ✅ DONE

- [x] `src/services/klingService.ts` - Gelöscht (komplett ungenutzt)
- [x] `Toolbar.tsx:692` - Ternary zu if/else geändert
- [x] `DockPanelContent.tsx:17` - Case block mit braces versehen

---

## Phase 2: TypeScript `any` Typen ersetzen

### Priorität: HOCH (häufig verwendete Dateien)

| Datei | Zeilen | Beschreibung |
|-------|--------|--------------|
| `src/components/common/RelinkDialog.tsx` | 7x any | MediaFile und Event Typen |
| `src/components/common/WelcomeOverlay.tsx` | 3x any | Event Handler Typen |
| `src/components/common/NativeHelperStatus.tsx` | 2x any | SystemInfo Extension |
| `src/components/mobile/MobileApp.tsx` | 3x any | Touch Event Typen |

### Priorität: MITTEL (Export/Engine)

| Datei | Zeilen | Beschreibung |
|-------|--------|--------------|
| `src/engine/export/FrameExporter.ts` | 5x any | Encoder Typen |
| `src/engine/ParallelDecodeManager.ts` | 4x any | MP4Box Typen |
| `src/components/export/ExportPanel.tsx` | 3x any | Settings Typen |

### Priorität: NIEDRIG (selten geändert)

| Datei | Zeilen | Beschreibung |
|-------|--------|--------------|
| `src/stores/mediaStore/slices/*.ts` | diverse | Store Action Typen |
| `src/services/projectSync.ts` | 2x any | Sync State Typen |

---

## Phase 3: React Hook Dependencies fixen

### useEffect Dependencies

| Datei | Zeile | Fehlende Deps |
|-------|-------|---------------|
| `src/App.tsx` | 121 | `showWelcome` bereits drin, aber Logik prüfen |
| `src/components/common/Toolbar.tsx` | 181 | `handleNew`, `handleOpen` |
| `src/components/export/ExportPanel.tsx` | 496 | `encoder`, `endExport`, `setExportProgress`, `startExport` |
| `src/components/export/FFmpegExportSection.tsx` | 233 | `isExporting` |

### useCallback Dependencies

| Datei | Zeile | Problem |
|-------|-------|---------|
| `src/components/preview/Preview.tsx` | 45 | `canvasRef` |

---

## Phase 4: Ungenutzte Variablen

| Datei | Variable | Aktion |
|-------|----------|--------|
| `src/engine/export/FrameExporter.ts` | `totalGpuTime`, `totalEncodeTime` | Entfernen oder verwenden |
| `src/engine/ParallelDecodeManager.ts` | `startSampleIndex` | Entfernen oder verwenden |
| `src/engine/ParallelDecodeManager.ts` | `MP4Box` import | Prüfen ob nötig |
| `src/stores/mediaStore/index.ts` | `compositions` | Prüfen ob nötig |

---

## Phase 5: Async Promise Executor

| Datei | Zeile | Problem |
|-------|-------|---------|
| `src/services/projectSync.ts` | ? | `new Promise(async (resolve) => ...)` |
| `src/engine/WebCodecsPlayer.ts` | ? | `new Promise(async (resolve) => ...)` |

**Fix:** Promise in async function wrappen statt async executor.

---

## Phase 6: React Compiler Warnings (Optional)

Diese sind schwieriger zu fixen und können später gemacht werden:

- 29x "Cannot access refs during render" - React Compiler spezifisch
- 12x "Calling setState in effect" - Meist false positives bei async calls

---

## Reihenfolge der Umsetzung

1. **Phase 2** - `any` Typen (verbessert Code-Qualität und IDE Support)
2. **Phase 3** - Hook Dependencies (verhindert Bugs)
3. **Phase 4** - Ungenutzte Variablen (Cleanup)
4. **Phase 5** - Async Promise (Best Practice)
5. **Phase 6** - Optional, wenn Zeit

---

## Notizen

- ESLint Config wurde bereits angepasst:
  - `any` ist jetzt nur noch Warning (nicht Error)
  - Unused vars mit `_` Präfix erlaubt
  - Empty catch blocks erlaubt

- TypeScript kompiliert ohne Fehler
- Build funktioniert
