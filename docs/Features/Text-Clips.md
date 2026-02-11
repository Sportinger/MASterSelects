# Text Clips

Text clips allow you to add text overlays directly to your timeline with full typography control. Text is rendered on the GPU, allowing all existing effects, blend modes, masks, and keyframe animations to work seamlessly.

## Creating Text Clips

### From Timeline Controls

1. Position the playhead where you want the text to appear
2. Click the **+ Text** button in the Timeline toolbar
3. A 5-second text clip appears on the topmost video track
4. Select the clip to edit text properties

### Text Clip Appearance

Text clips display with a distinctive purple gradient to differentiate them from video/image clips. The clip shows:
- A "T" icon indicator
- Preview of the first 30 characters of text content
- Duration display

## Text Properties Panel

When a text clip is selected, the Properties Panel shows a **Text** tab with full typography controls:

### Content Section
- **Textarea**: Multi-line text input
- Live preview updates as you type

### Font Section

| Control | Description | Range |
|---------|-------------|-------|
| **Family** | Google Font selection | 50 fonts available |
| **Size** | Font size in pixels | 8-500px |
| **Weight** | Font weight | 100 (Thin) to 900 (Black) |
| **Style** | Normal or Italic | Normal / Italic |

### Color Section
- **Fill**: Color picker for text fill color
- Supports hex color input

### Alignment Section

| Alignment | Options |
|-----------|---------|
| **Horizontal** | Left (L), Center (C), Right (R) |
| **Vertical** | Top (T), Middle (M), Bottom (B) |

### Spacing Section

| Control | Description | Range |
|---------|-------------|-------|
| **Line Height** | Line spacing multiplier | 0.5 - 3.0 |
| **Letter Spacing** | Space between characters | -10 to 50px |

### Stroke (Outline) Section

Enable with checkbox, then configure:

| Control | Description | Range |
|---------|-------------|-------|
| **Color** | Stroke color | Color picker |
| **Width** | Stroke width | 1-20px |

### Shadow Section

Enable with checkbox, then configure:

| Control | Description | Range |
|---------|-------------|-------|
| **Color** | Shadow color | Color picker |
| **Offset X** | Horizontal offset | -50 to 50px |
| **Offset Y** | Vertical offset | -50 to 50px |
| **Blur** | Shadow blur radius | 0-50px |

### Preview Section
Live font preview showing current text with all styling applied (scaled down).

## Available Fonts

The 50 most popular Google Fonts are available, organized by category:

### Sans-Serif
Roboto, Open Sans, Lato, Montserrat, Poppins, Inter, Nunito, Work Sans, Rubik, Source Sans Pro, Raleway, Oswald, Nunito Sans, Josefin Sans, Quicksand, Mukta, Barlow, Ubuntu, Archivo, Karla

### Serif
Playfair Display, Lora, Merriweather, PT Serif, Noto Serif, Libre Baskerville, Crimson Text, Cormorant Garamond, EB Garamond

### Display
Bebas Neue, Anton, Righteous, Lobster, Permanent Marker, Abril Fatface, Pacifico

### Handwriting
Dancing Script, Caveat, Satisfy, Great Vibes, Indie Flower, Kaushan Script, Sacramento, Cookie

### Monospace
Fira Code, Source Code Pro, JetBrains Mono, Roboto Mono, Inconsolata

## GPU Rendering

Text clips use a hybrid rendering approach:

1. **Canvas2D Rendering**: Text is rendered to a 1920x1080 canvas using the Canvas2D API
2. **GPU Texture**: The canvas is imported as a GPU texture
3. **Compositing**: Text is composited like any other layer in the WebGPU pipeline

This approach means:
- All 37 blend modes work with text
- All 9 GPU effects can be applied
- Masks work normally (including feathered masks)
- Transform animations (position, scale, rotation) work
- Keyframe animation is fully supported

## Animation

Text clips support all standard transform animations:

| Property | Keyframeable | Notes |
|----------|--------------|-------|
| Position X/Y/Z | Yes | Move text across screen |
| Scale X/Y | Yes | Grow/shrink text |
| Rotation X/Y/Z | Yes | Spin text in 3D |
| Opacity | Yes | Fade text in/out |

Effect animations also work:
- Add blur for fade-in effects
- Use hue shift for color cycling
- Apply pixelate for reveal effects

## Serialization

Text clips are fully persisted when saving projects:
- Text content and all typography properties are saved
- Font is reloaded from Google Fonts on project open
- Canvas is re-rendered on restore

## Technical Implementation

### Files

| File | Purpose |
|------|---------|
| `src/services/googleFontsService.ts` | Font loading via CSS injection |
| `src/services/textRenderer.ts` | Canvas2D text rendering |
| `src/components/panels/TextTab.tsx` | Properties panel UI |
| `src/stores/timeline/clipSlice.ts` | addTextClip, updateTextProperties |

### TextClipProperties Type

```typescript
interface TextClipProperties {
  text: string;
  fontFamily: string;
  fontSize: number;
  fontWeight: number;
  fontStyle: 'normal' | 'italic';
  color: string;
  textAlign: 'left' | 'center' | 'right';
  verticalAlign: 'top' | 'middle' | 'bottom';
  lineHeight: number;
  letterSpacing: number;
  strokeEnabled: boolean;
  strokeColor: string;
  strokeWidth: number;
  shadowEnabled: boolean;
  shadowColor: string;
  shadowOffsetX: number;
  shadowOffsetY: number;
  shadowBlur: number;
  pathEnabled: boolean;
  pathPoints: PathPoint[];
}
```

---

## Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| [`clipSlice.test.ts`](../../tests/stores/timeline/clipSlice.test.ts) | 104 | Text clip operations (included in clip tests) |

Run tests: `npx vitest run`

---

## Future Enhancements

Planned but not yet implemented:
- Text on bezier path (curved text)
- Per-character animation
- Text presets/styles library
- Gradient fills
- Multiple shadows
- Background boxes
