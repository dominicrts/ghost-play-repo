// src/onnx/pigtSession.ts
import * as ort from "onnxruntime-web";
import type { InferenceSession, Tensor } from "onnxruntime-web";

let sessionPromise: Promise<InferenceSession> | null = null;

export async function createPIGTSession(modelUrl: string): Promise<InferenceSession> {
  if (sessionPromise) return sessionPromise;

  const webgpuAvailable =
    typeof navigator !== "undefined" && !!(navigator as any).gpu;

  if (webgpuAvailable) {
    // Switch to WebGPU-backed bundle (recommended by recent guides)
    // import alias via Vite config or dynamic import if needed. 
    // @ts-ignore
    const ortWebGPU = await import("onnxruntime-web/webgpu");
    // propagate env & types
    (ort as any).env = ortWebGPU.env;
    (ort as any).InferenceSession = ortWebGPU.InferenceSession;
  } else {
    // WASM backend with SIMD + threads as far as browser allows
    ort.env.wasm.simd = true;
    // threads > 1 requires COOP/COEP headers; keep modest default
    ort.env.wasm.numThreads = 2;
  }

  const executionProviders = webgpuAvailable ? ["webgpu", "wasm"] : ["wasm"];

  sessionPromise = ort.InferenceSession.create(modelUrl, {
    executionProviders,
  });

  return sessionPromise;
}

export type OrtTensor = Tensor;
export { ort };
