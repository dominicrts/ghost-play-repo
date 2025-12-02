// src/three/GhostMaterial.tsx
import { useMemo } from "react";
import * as THREE from "three";
import { extend, ReactThreeFiber } from "@react-three/fiber";

class GhostShaderMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uColor: { value: new THREE.Color(0x7fffd4) },
        uIntensity: { value: 1.0 },
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 worldPos = modelMatrix * vec4(position, 1.0);
          vWorldPos = worldPos.xyz;
          gl_Position = projectionMatrix * viewMatrix * worldPos;
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        uniform vec3 uColor;
        uniform float uIntensity;

        void main() {
          // simple fresnel-style rim
          vec3 viewDir = normalize(cameraPosition - vWorldPos);
          float fresnel = pow(1.0 - max(dot(viewDir, normalize(vNormal)), 0.0), 2.0);
          float alpha = clamp(fresnel * 1.5 * uIntensity, 0.0, 1.0);
          gl_FragColor = vec4(uColor, alpha);
        }
      `,
    });
  }
}

extend({ GhostShaderMaterial });

declare global {
  namespace JSX {
    interface IntrinsicElements {
      ghostShaderMaterial: ReactThreeFiber.Object3DNode<
        GhostShaderMaterial,
        typeof GhostShaderMaterial
      >;
    }
  }
}

export function useGhostMaterial(intensity = 1.0, color = 0x7fffd4) {
  return useMemo(() => {
    const mat = new GhostShaderMaterial();
    mat.uniforms.uIntensity.value = intensity;
    mat.uniforms.uColor.value = new THREE.Color(color);
    return mat;
  }, [intensity, color]);
}
