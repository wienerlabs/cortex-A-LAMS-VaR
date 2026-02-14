/**
 * Vitest Setup File
 *
 * Polyfills for Node.js compatibility with older libraries.
 * This runs before each test file.
 */

// Load environment variables from .env file
import 'dotenv/config';

import buffer from 'buffer';

// Polyfill buffer.SlowBuffer for avsc library (Solend SDK dependency)
// SlowBuffer was deprecated in Node.js but avsc (version used by Solend) still uses it
// This is a compatibility fix, not a mock
if (!(buffer as any).SlowBuffer) {
  Object.defineProperty(buffer, 'SlowBuffer', {
    value: buffer.Buffer,
    writable: false,
    configurable: false,
  });
}

