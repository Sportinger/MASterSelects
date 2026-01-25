/**
 * Google Fonts Service - Dynamically loads fonts from Google Fonts
 * Uses the FontFace API for efficient font loading
 */

import { Logger } from './logger';

const log = Logger.create('GoogleFontsService');

export interface FontConfig {
  family: string;
  weights: number[];
  category: 'sans-serif' | 'serif' | 'display' | 'handwriting' | 'monospace';
}

// Top 50 most popular Google Fonts
export const POPULAR_FONTS: FontConfig[] = [
  // Sans-serif fonts
  { family: 'Roboto', weights: [100, 300, 400, 500, 700, 900], category: 'sans-serif' },
  { family: 'Open Sans', weights: [300, 400, 500, 600, 700, 800], category: 'sans-serif' },
  { family: 'Lato', weights: [100, 300, 400, 700, 900], category: 'sans-serif' },
  { family: 'Montserrat', weights: [100, 200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Poppins', weights: [100, 200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Inter', weights: [100, 200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Oswald', weights: [200, 300, 400, 500, 600, 700], category: 'sans-serif' },
  { family: 'Raleway', weights: [100, 200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Nunito', weights: [200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Ubuntu', weights: [300, 400, 500, 700], category: 'sans-serif' },
  { family: 'Rubik', weights: [300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Work Sans', weights: [100, 200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Nunito Sans', weights: [200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Quicksand', weights: [300, 400, 500, 600, 700], category: 'sans-serif' },
  { family: 'Mulish', weights: [200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Barlow', weights: [100, 200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'Manrope', weights: [200, 300, 400, 500, 600, 700, 800], category: 'sans-serif' },
  { family: 'IBM Plex Sans', weights: [100, 200, 300, 400, 500, 600, 700], category: 'sans-serif' },
  { family: 'Source Sans 3', weights: [200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },
  { family: 'DM Sans', weights: [100, 200, 300, 400, 500, 600, 700, 800, 900], category: 'sans-serif' },

  // Serif fonts
  { family: 'Playfair Display', weights: [400, 500, 600, 700, 800, 900], category: 'serif' },
  { family: 'Merriweather', weights: [300, 400, 700, 900], category: 'serif' },
  { family: 'Lora', weights: [400, 500, 600, 700], category: 'serif' },
  { family: 'PT Serif', weights: [400, 700], category: 'serif' },
  { family: 'Libre Baskerville', weights: [400, 700], category: 'serif' },
  { family: 'Bitter', weights: [100, 200, 300, 400, 500, 600, 700, 800, 900], category: 'serif' },
  { family: 'EB Garamond', weights: [400, 500, 600, 700, 800], category: 'serif' },
  { family: 'Crimson Text', weights: [400, 600, 700], category: 'serif' },
  { family: 'Cormorant Garamond', weights: [300, 400, 500, 600, 700], category: 'serif' },
  { family: 'Libre Caslon Text', weights: [400, 700], category: 'serif' },

  // Display fonts
  { family: 'Bebas Neue', weights: [400], category: 'display' },
  { family: 'Anton', weights: [400], category: 'display' },
  { family: 'Archivo Black', weights: [400], category: 'display' },
  { family: 'Righteous', weights: [400], category: 'display' },
  { family: 'Alfa Slab One', weights: [400], category: 'display' },
  { family: 'Bungee', weights: [400], category: 'display' },
  { family: 'Abril Fatface', weights: [400], category: 'display' },
  { family: 'Fredoka One', weights: [400], category: 'display' },
  { family: 'Titan One', weights: [400], category: 'display' },
  { family: 'Permanent Marker', weights: [400], category: 'display' },

  // Handwriting fonts
  { family: 'Dancing Script', weights: [400, 500, 600, 700], category: 'handwriting' },
  { family: 'Pacifico', weights: [400], category: 'handwriting' },
  { family: 'Caveat', weights: [400, 500, 600, 700], category: 'handwriting' },
  { family: 'Great Vibes', weights: [400], category: 'handwriting' },
  { family: 'Sacramento', weights: [400], category: 'handwriting' },

  // Monospace fonts
  { family: 'Roboto Mono', weights: [100, 200, 300, 400, 500, 600, 700], category: 'monospace' },
  { family: 'Source Code Pro', weights: [200, 300, 400, 500, 600, 700, 800, 900], category: 'monospace' },
  { family: 'Fira Code', weights: [300, 400, 500, 600, 700], category: 'monospace' },
  { family: 'JetBrains Mono', weights: [100, 200, 300, 400, 500, 600, 700, 800], category: 'monospace' },
  { family: 'Space Mono', weights: [400, 700], category: 'monospace' },
];

class GoogleFontsService {
  private loadedFonts = new Set<string>();
  private loadingPromises = new Map<string, Promise<void>>();
  private cssLoaded = new Set<string>();

  /**
   * Load a font with specific weight using CSS link injection
   * This is more reliable than FontFace API for Google Fonts
   */
  async loadFont(family: string, weight: number = 400): Promise<void> {
    const key = `${family}-${weight}`;

    // Already loaded
    if (this.loadedFonts.has(key)) {
      return;
    }

    // Currently loading - wait for existing promise
    if (this.loadingPromises.has(key)) {
      return this.loadingPromises.get(key);
    }

    const promise = this.doLoadFont(family, weight);
    this.loadingPromises.set(key, promise);

    try {
      await promise;
      this.loadedFonts.add(key);
    } catch (e) {
      log.warn(`Failed to load ${family} ${weight}`, e);
    } finally {
      this.loadingPromises.delete(key);
    }
  }

  private async doLoadFont(family: string, weight: number): Promise<void> {
    // Use CSS link for this specific font weight
    const cssKey = `${family}:wght@${weight}`;

    if (!this.cssLoaded.has(cssKey)) {
      // Create link element to load the font CSS
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = `https://fonts.googleapis.com/css2?family=${encodeURIComponent(family)}:wght@${weight}&display=swap`;
      document.head.appendChild(link);
      this.cssLoaded.add(cssKey);
    }

    // Wait for the font to actually be loaded
    await document.fonts.load(`${weight} 16px "${family}"`);

    // Double-check with ready promise
    await document.fonts.ready;
  }

  /**
   * Preload common weights for a font family
   */
  async preloadFont(family: string): Promise<void> {
    const fontConfig = POPULAR_FONTS.find(f => f.family === family);
    if (!fontConfig) {
      // Load default weight if not in our list
      await this.loadFont(family, 400);
      return;
    }

    // Load regular weight first (fastest perceived load)
    if (fontConfig.weights.includes(400)) {
      await this.loadFont(family, 400);
    }

    // Then load other common weights in background
    const otherWeights = fontConfig.weights.filter(w => w !== 400);
    Promise.all(otherWeights.map(w => this.loadFont(family, w))).catch(() => {});
  }

  /**
   * Get all available font families
   */
  getAvailableFonts(): FontConfig[] {
    return POPULAR_FONTS;
  }

  /**
   * Get fonts by category
   */
  getFontsByCategory(category: FontConfig['category']): FontConfig[] {
    return POPULAR_FONTS.filter(f => f.category === category);
  }

  /**
   * Check if a specific font weight is loaded
   */
  isFontLoaded(family: string, weight: number = 400): boolean {
    return this.loadedFonts.has(`${family}-${weight}`);
  }

  /**
   * Get available weights for a font family
   */
  getAvailableWeights(family: string): number[] {
    const fontConfig = POPULAR_FONTS.find(f => f.family === family);
    return fontConfig?.weights || [400];
  }

  /**
   * Get CSS font-family string with fallback
   */
  getFontFamilyCSS(family: string): string {
    const fontConfig = POPULAR_FONTS.find(f => f.family === family);
    const fallback = fontConfig?.category || 'sans-serif';
    return `"${family}", ${fallback}`;
  }
}

export const googleFontsService = new GoogleFontsService();
