FROM node:20-alpine
WORKDIR /app

# curl for the compose healthcheck / debugging
RUN apk add --no-cache curl

COPY package.json ./
RUN npm install --omit=dev

COPY src ./src

EXPOSE 4100
CMD ["node", "src/index.js"]
