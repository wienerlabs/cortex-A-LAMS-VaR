import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    // Exclude heavy integration tests that require real blockchain/wallet
    // Run these manually: npx vitest run src/services/__tests__/portfolioIntegration.test.ts
    exclude: [
      '**/node_modules/**',
      '**/portfolioIntegration.test.ts',
    ],
    environment: 'node',
    testTimeout: 30000,
    setupFiles: ['./vitest.setup.ts'],
    // Use threads pool for better memory management
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: true,
      },
    },
  },
});

