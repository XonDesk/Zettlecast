# The web UI in OrbStack. Sign-in settings come from frontend/.env.local at run time (env_file).
FROM node:24-slim

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
RUN npm run build

ENV NODE_ENV=production PORT=3100
EXPOSE 3100
CMD ["sh", "-c", "exec npx next start -H 0.0.0.0 -p ${PORT}"]
