import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API runs on :8010; proxy /v1 so the browser talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/v1": { target: "http://127.0.0.1:8010", changeOrigin: true } } },
});
