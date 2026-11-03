import { defineConfig } from 'vite'

export default defineConfig({
  // During development the page talks to the engine server on port 3021.
  server: { port: 5173, proxy: { '/engine': { target: 'ws://localhost:3021', ws: true } } },
  build: { outDir: 'dist' },
})
