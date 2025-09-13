import { defineConfig } from 'vite'
import path from 'path';

// https://vite.dev/config/
export default defineConfig({
  build: {
    target: 'node18',
    lib: {
      entry: path.resolve(__dirname, 'src/index.ts'), // Entry point file
      name: '@meaningfully/core',
      fileName: (format) => `meaningfully-core.${format}.js`,
      formats: ['es', 'cjs']
    },
    rollupOptions: {
      external: [
        'better-sqlite3', 
        'path', 
        'fs', 
        'os', 
        'http', 
        'https', 
        'url', 
        'stream', 
        'util', 
        'events',
        'crypto',
        'buffer',
        'querystring',
        'zlib',
        'net',
        'tls',
        'child_process'
      ],
      output: {
        globals: {
          'better-sqlite3': 'Database'
        }
      }
    }
  }  
})
