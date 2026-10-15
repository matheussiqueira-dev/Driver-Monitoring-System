import type { Point } from "../domain/geometry";

export interface MediaSize {
  readonly width: number;
  readonly height: number;
}

export interface Viewport {
  readonly width: number;
  readonly height: number;
  readonly dpr: number;
}

interface Mapping {
  readonly sourceWidth: number;
  readonly sourceHeight: number;
  readonly offsetX: number;
  readonly offsetY: number;
  readonly scale: number;
  readonly mirrored: boolean;
}

/**
 * Maps normalized landmarks (0..1 of the source frame) to CSS pixels of an
 * element rendered with `object-fit: cover`, optionally mirrored.
 */
export class CoordinateMapper {
  private mapping: Mapping = { sourceWidth: 1, sourceHeight: 1, offsetX: 0, offsetY: 0, scale: 1, mirrored: false };
  private viewport: Viewport = { width: 1, height: 1, dpr: 1 };

  update(viewport: Viewport, source: MediaSize, mirrored: boolean): void {
    const width = Math.max(1, viewport.width);
    const height = Math.max(1, viewport.height);
    const sourceWidth = source.width > 0 ? source.width : width;
    const sourceHeight = source.height > 0 ? source.height : height;
    const scale = Math.max(width / sourceWidth, height / sourceHeight);

    this.viewport = { width, height, dpr: viewport.dpr > 0 ? viewport.dpr : 1 };
    this.mapping = {
      sourceWidth,
      sourceHeight,
      offsetX: (width - sourceWidth * scale) / 2,
      offsetY: (height - sourceHeight * scale) / 2,
      scale,
      mirrored,
    };
  }

  get size(): Viewport {
    return this.viewport;
  }

  get isMirrored(): boolean {
    return this.mapping.mirrored;
  }

  toCanvasPoint(landmark: Point): Point {
    const { sourceWidth, sourceHeight, offsetX, offsetY, scale, mirrored } = this.mapping;
    const nx = mirrored ? 1 - landmark.x : landmark.x;
    return {
      x: offsetX + nx * sourceWidth * scale,
      y: offsetY + landmark.y * sourceHeight * scale,
      z: (landmark.z ?? 0) * sourceWidth * scale,
    };
  }
}
