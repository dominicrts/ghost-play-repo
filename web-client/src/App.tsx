// src/App.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stats } from "@react-three/drei";
import { motion } from "framer-motion";
import { GhostField } from "./three/GhostField";

const N = 22;
const T_future = 20;
const dModel = 128;
const goalDim = 16;
const posDim = 2;

type WorkerResult = {
  type: "RUN_RESULT";
  payload: {
    T_future: number;
    N: number;
    posDim: number;
    posSeq: ArrayBuffer; // [1, T_future, N, posDim]
  };
};

export const App: React.FC = () => {
  const [workerReady, setWorkerReady] = useState(false);
  const [ghostTraj, setGhostTraj] = useState<Float32Array | null>(null);

  // Simple initial player positions
  const [currentPositions, setCurrentPositions] = useState<Float32Array>(() => {
    const arr = new Float32Array(N * posDim);
    for (let i = 0; i < N; i++) {
      arr[i * 2 + 0] = -20 + i; // x
      arr[i * 2 + 1] = 0;       // y
    }
    return arr;
  });

  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("./workers/pigtWorker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<any>) => {
      const data = event.data;
      if (data.type === "INIT_DONE") {
        setWorkerReady(true);
      } else if (data.type === "RUN_RESULT") {
        const msg = data as WorkerResult;
        const { T_future, N, posDim, posSeq } = msg.payload;
        const arr = new Float32Array(posSeq); // length = 1 * T_future * N * posDim
        // We can pass this directly to GhostField; it expects [T_future, N, 2] flattened
        setGhostTraj(arr);
      }
    };

    worker.postMessage({
      type: "INIT",
      modelUrl: "/models/pigt_web_int8.onnx",
      config: { T_future, N, dModel, goalDim, posDim },
    });

    return () => {
      worker.terminate();
    };
  }, []);

  // Simple goal tensor (first two dims = target offset)
  const goal = useMemo(() => new Float32Array(goalDim), []);

  const runSimulation = (modifiedGoal: Float32Array) => {
    if (!workerRef.current || !workerReady) return;

    // For now we create dummy memory with fixed length L_mem.
    // This must match how you exported: L_mem = T_in * N; we used T_in=10 in Python.
    const T_in = 10;
    const L_mem = T_in * N;

    const memory = new Float32Array(L_mem * dModel);
    const memoryRoles = new Int32Array(L_mem);
    const roleIds = new Int32Array(N);
    const tgtInit = new Float32Array(T_future * N * dModel);
    const globalGoal = modifiedGoal;

    // Last two positions (for now: same as current positions)
    const posLast = new Float32Array(N * posDim);
    const posPrev = new Float32Array(N * posDim);
    posLast.set(currentPositions);
    posPrev.set(currentPositions); // zero initial velocity for now

    // Simple role assignments: cycle through 0..numRoles-1
    const numRoles = 8;
    for (let i = 0; i < N; i++) {
      roleIds[i] = i % numRoles;
    }
    for (let i = 0; i < L_mem; i++) {
      memoryRoles[i] = i % numRoles;
    }

    workerRef.current.postMessage({
      type: "RUN",
      payload: {
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
      },
    });
  };

  // Drag callback: encode drag position into goal tensor
  const handleDrag = (_event: any, info: { point: { x: number; y: number } }) => {
    const gx = info.point.x / 200;
    const gy = info.point.y / 200;
    goal[0] = gx;
    goal[1] = gy;
    runSimulation(goal);
  };

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#020617" }}>
      <Canvas camera={{ position: [0, 40, 80], fov: 40 }}>
        <ambientLight intensity={0.4} />
        <directionalLight position={[20, 50, 20]} intensity={1.0} castShadow />
        <GhostField
          N={N}
          currentPositions={currentPositions}
          ghostTraj={ghostTraj}
          T_future={T_future}
        />
        <OrbitControls />
        <Stats />
      </Canvas>

      {/* Draggable WR target (Framer Motion) */}
      <motion.div
        drag
        dragMomentum={false}
        style={{
          position: "absolute",
          bottom: 40,
          left: "50%",
          width: 40,
          height: 40,
          borderRadius: 999,
          marginLeft: -20,
          background: "#38bdf8",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 12,
          color: "#0f172a",
          cursor: "grab",
          boxShadow: "0 0 20px rgba(56,189,248,0.7)",
        }}
        onDrag={handleDrag}
      >
        WR
      </motion.div>

      {!workerReady && (
        <div
          style={{
            position: "absolute",
            top: 16,
            left: 16,
            padding: "4px 8px",
            background: "#0f172a",
            color: "#e5e7eb",
            borderRadius: 8,
            fontSize: 12,
          }}
        >
          Loading Ghost Play model…
        </div>
      )}
    </div>
  );
};
