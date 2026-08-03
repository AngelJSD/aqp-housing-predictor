import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Minimal, self-contained production output for the Docker image — see
  // node_modules/next/dist/docs/.../output.md.
  output: "standalone",
};

export default nextConfig;
