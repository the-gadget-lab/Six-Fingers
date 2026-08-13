export interface ModelSpec {
  file: string;
  inputSize: number;
  resizeSize: number;
  mean: [number, number, number];
  std: [number, number, number];
  threshold: number;
}

export interface ScoreRequest {
  kind: "score";
  url: string;
}

export interface ScoreResponse {
  ok: boolean;
  prob?: number;
  error?: string;
}

export interface Verdict {
  prob: number;
  isAi: boolean;
}

export const MIN_IMAGE_DIM = 96;
