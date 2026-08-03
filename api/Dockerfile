# node:*-alpine confirmed broken for this image: onnxruntime-node's
# prebuilt native binary needs glibc's dynamic linker, which musl
# (Alpine) doesn't have — ERR_DLOPEN_FAILED, "ld-linux-aarch64.so.1: No
# such file or directory". node:*-slim (Debian) works.
FROM node:22-slim
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install --omit=dev

COPY src/ ./src/

ENV PORT=3000
EXPOSE 3000

CMD ["node", "src/server.js"]
