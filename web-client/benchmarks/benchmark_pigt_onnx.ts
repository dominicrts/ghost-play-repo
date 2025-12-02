// web-client/benchmarks/benchmark_pigt_onnx.ts
import * as ort from "onnxruntime-web";

async function main() {
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = 2;

  const session = await ort.InferenceSession.create(
    "./public/models/pigt_int8.onnx",
    { executionProviders: ["wasm"] }
  );

  const T_in = 10;
  const T_future = 20;
  const N = 22;
  const inChannels = 64;
  const dModel = 128;
  const numRoles = 8;
  const goalDim = 16;

  const xPast = new Float32Array(T_in * N * inChannels);
  const posPast = new Float32Array(T_in * N * 2);
  const roleIds = new Int32Array(N);
  const tgtInit = new Float32Array(T_future * N * dModel);
  const globalGoal = new Float32Array(goalDim);

  const feeds: Record<string, ort.Tensor> = {
    x_past: new ort.Tensor("float32", xPast, [1, T_in, N, inChannels]),
    pos_past: new ort.Tensor("float32", posPast, [1, T_in, N, 2]),
    role_ids: new ort.Tensor("int32", roleIds, [1, N]),
    tgt_init: new ort.Tensor("float32", tgtInit, [1, T_future, N, dModel]),
    global_goal: new ort.Tensor("float32", globalGoal, [1, goalDim]),
  };

  // Warmup
  await session.run(feeds);

  const runs = 50;
  const times: number[] = [];

  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await session.run(feeds);
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  const avg = times.reduce((a, b) => a + b, 0) / runs;
  const p95 = times.slice().sort((a, b) => a - b)[Math.floor(0.95 * runs)];

  console.log(`Average latency: ${avg.toFixed(2)} ms`);
  console.log(`95th percentile: ${p95.toFixed(2)} ms`);
  console.log(`Target per-frame budget (60fps): 16.67 ms`);

  if (p95 < 16.67) {
    console.log("✅ Theoretical 60fps budget satisfied on this machine.");
  } else {
    console.log("⚠️ Consider further quantization or reducing model size.");
  }
}

main().catch(console.error);
