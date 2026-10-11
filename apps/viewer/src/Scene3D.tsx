import { Canvas } from "@react-three/fiber";
import { Grid, Line, OrbitControls } from "@react-three/drei";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { Robot, Track } from "./api";

type Props = {
  voxels: { xyz: Float32Array; size: number } | null;
  points: { xyz: Float32Array; rgb: Uint8Array } | null;
  tracks: Track[];
  path: number[][] | null;
  T_map_base: number[][] | null;
  robot: Robot;
  floorZ: number;
  show: { voxels: boolean; points: boolean; tracks: boolean; ground: boolean };
};

function Voxels({ xyz, size, floorZ }: { xyz: Float32Array; size: number; floorZ: number }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const n = xyz.length / 3;
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    const mat = new THREE.Matrix4();
    const col = new THREE.Color();
    for (let i = 0; i < n; i++) {
      mat.makeTranslation(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
      m.setMatrixAt(i, mat);
      const h = Math.min(Math.max((xyz[3 * i + 2] - floorZ) / 1.5, 0), 1);
      col.setHSL(0.62 - 0.62 * h, 0.65, 0.5);
      m.setColorAt(i, col);
    }
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }, [xyz, n, floorZ]);
  return (
    <instancedMesh ref={ref} args={[undefined, undefined, n]} key={n}>
      <boxGeometry args={[size * 0.95, size * 0.95, size * 0.95]} />
      <meshLambertMaterial />
    </instancedMesh>
  );
}

function Points({ xyz, rgb }: { xyz: Float32Array; rgb: Uint8Array }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
    g.setAttribute("color", new THREE.BufferAttribute(rgb, 3, true));
    return g;
  }, [xyz, rgb]);
  return (
    <points geometry={geom}>
      <pointsMaterial size={0.02} vertexColors />
    </points>
  );
}

// Data are in the map frame (z up). Three.js is y up: we rotate the whole scene for DISPLAY ONLY.
export function Scene3D({ voxels, points, tracks, path, T_map_base, robot, floorZ, show }: Props) {
  const robotPos = T_map_base ? [T_map_base[0][3], T_map_base[1][3], T_map_base[2][3]] : null;
  const yaw = T_map_base ? Math.atan2(T_map_base[1][0], T_map_base[0][0]) : 0;
  return (
    <Canvas camera={{ position: [3, 7, 6], fov: 50, up: [0, 1, 0] }}>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} />
      <group rotation={[-Math.PI / 2, 0, 0]}>
        {show.ground && (
          <Grid
            args={[20, 20]}
            position={[3, 2, floorZ - 0.001]}
            rotation={[Math.PI / 2, 0, 0]}
            cellSize={0.5}
            sectionSize={1}
            infiniteGrid={false}
          />
        )}
        {show.voxels && voxels && voxels.xyz.length > 0 && <Voxels xyz={voxels.xyz} size={voxels.size} floorZ={floorZ} />}
        {show.points && points && <Points xyz={points.xyz} rgb={points.rgb} />}
        {show.tracks &&
          tracks.map((t) => (
            <mesh key={t.track_id} position={t.center_m as [number, number, number]}>
              <boxGeometry args={t.visible_extent_m as [number, number, number]} />
              <meshBasicMaterial wireframe color={t.lifecycle === "confirmed" ? "#e8590c" : "#868e96"} />
            </mesh>
          ))}
        {path && path.length > 1 && (
          <Line points={path.map((p) => [p[0], p[1], floorZ + 0.02] as [number, number, number])} color="#e03131" lineWidth={3} />
        )}
        {robotPos && (
          <group position={robotPos as [number, number, number]} rotation={[0, 0, yaw]}>
            <mesh position={[0, 0, robot.height_m / 2]} rotation={[Math.PI / 2, 0, 0]}>
              <cylinderGeometry args={[robot.radius_m, robot.radius_m, robot.height_m, 32]} />
              <meshLambertMaterial color="#7048e8" transparent opacity={0.45} />
            </mesh>
            <mesh position={[robot.radius_m, 0, robot.height_m]}>
              <boxGeometry args={[0.12, 0.04, 0.04]} />
              <meshBasicMaterial color="#ffd43b" />
            </mesh>
          </group>
        )}
      </group>
      <OrbitControls target={[3, 0, -2]} makeDefault />
    </Canvas>
  );
}
