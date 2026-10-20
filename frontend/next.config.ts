import { networkInterfaces } from "os";
import type { NextConfig } from "next";

// This computer's network addresses: the dev server only serves its scripts to pages opened
// from an allowed origin (a phone uses http://<address>:3000)
const lanAddresses = Object.values(networkInterfaces())
  .flat()
  .filter((a) => a && a.family === "IPv4" && !a.internal)
  .map((a) => a!.address);

const nextConfig: NextConfig = {
  allowedDevOrigins: lanAddresses,
  experimental: {
    // uploads (podcast audio) pass through the sign-in check in proxy.ts
    proxyClientMaxBodySize: "2gb",
  },
};

export default nextConfig;
