# Alpine confirmed fine here (unlike api/, which needs onnxruntime-node's
# glibc-only native binary) — pg and Recharts are pure JS, tested directly
# before choosing this base image.
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# output: standalone (next.config.ts) traces only the files actually
# needed, so this final stage doesn't need node_modules installed at all.
FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

ENV PORT=3000
EXPOSE 3000
CMD ["node", "server.js"]
