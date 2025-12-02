/// <reference lib="webworker" />
import { createPIGTSession, ort, OrtTensor } from "../onnx/pigtSession";

type InitMessage = {
  type: "INIT";
  modelUrl: string;

  // Optional: allow passing precomputed memory from main thread later
  config: {
    T_future: number;
    N: number;
    dModel: number;
    goalDim: number;
    posDim: number;
  };
};

type RunMessage = {
  type: "RUN";
  payload: {
    // Shapes must match ONNX graph
    T_future: number;
    N: number;
    dModel: number;
    goalDim: number;
    posDim: number;

    // Precomputed encoder outputs (or dummy for now)
    memory: Float32Array;       // [1, L_mem, D]
    memoryRoles: Int32Array;    // [1, L_mem]
    roleIds: Int32Array;        // [1, N]

    // Decoder init & goal
    tgtInit: Float32Array;      // [1, T_future, N, D]
    globalGoal: Float32Array;   // [1, goalDim]

    // Last two positions from past frames
    posLast: Float32Array;      // [1, N, posDim]
    posPrev: Float32Array;      // [1, N, posDim]
  };
};

type WorkerMessage = InitMessage | RunMessage;

let sessionReady: Promise<ort.InferenceSession> | null = null;

self.addEventListener("message", async (event: MessageEvent<WorkerMessage>) => {
  const data = event.data;

  if (data.type === "INIT") {
    const { modelUrl } = data;
    sessionReady = createPIGTSession(modelUrl);
    const session = await sessionReady;
    (self as any).postMessage({
      type: "INIT_DONE",
      inputNames: session.inputNames,
      outputNames: session.outputNames,
    });
    return;
  }

  if (data.type === "RUN") {
    if (!sessionReady) {
      console.error("PIGT session not initialized");
      return;
    }
    const session = await sessionReady;
    const {
      T_future,
      N,
      dModel,
      goalDim,
      posDim,
      memory,
      memoryRoles,
      roleIds,
      tgtInit,
      globalGoal,
      posLast,
      posPrev,
    } = data.payload;

    // Infer L_mem from memory length
    const L_mem = memory.length / (1 * dModel); // B=1
    if (!Number.isInteger(L_mem)) {
      console.error("Invalid memory length vs dModel");
      return;
    }

    const feeds: Record<string, OrtTensor> = {};

    feeds["memory"] = new ort.Tensor("float32", memory, [1, L_mem, dModel]);
    feeds["memory_roles"] = new ort.Tensor("int32", memoryRoles, [1, L_mem]);
    feeds["role_ids"] = new ort.Tensor("int32", roleIds, [1, N]);
    feeds["tgt_init"] = new ort.Tensor("float32", tgtInit, [1, T_future, N, dModel]);
    feeds["global_goal"] = new ort.Tensor("float32", globalGoal, [1, goalDim]);
    feeds["pos_last"] = new ort.Tensor("float32", posLast, [1, N, posDim]);
    feeds["pos_prev"] = new ort.Tensor("float32", posPrev, [1, N, posDim]);

    const results = await session.run(feeds);
    const posSeq = results["pos_seq"] as OrtTensor; // [1, T_future, N, posDim]

    const posData = posSeq.data as Float32Array;

    (self as any).postMessage(
      {
        type: "RUN_RESULT",
        payload: {
          T_future,
          N,
          posDim,
          posSeq: posData.buffer, // [1, T_future, N, posDim] flattened
        },
      },
      [posData.buffer]
    );
  }
});
