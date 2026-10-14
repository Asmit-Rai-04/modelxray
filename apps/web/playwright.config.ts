import { defineConfig } from "@playwright/test";
import path from "node:path";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  timeout: 180000,
  expect: { timeout: 15000 },
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    actionTimeout: 15000,
    baseURL: "http://localhost:59300",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "frontend", testMatch: /workspace\.spec\.ts/ },
    { name: "live-api", testMatch: /live\.spec\.ts/ },
  ],
  webServer: [
    {
      command: "npm run start -- -p 59300",
      url: "http://localhost:59300",
      reuseExistingServer: false,
      timeout: 120000,
    },
    {
      command:
        "python -m uvicorn modelxray.main:app --host 127.0.0.1 --port 8001",
      cwd: path.resolve(".test-artifacts"),
      env: { PYTHONPATH: path.resolve("../../apps/api") },
      url: "http://127.0.0.1:8001/health",
      reuseExistingServer: false,
      timeout: 60000,
    },
  ],
});
