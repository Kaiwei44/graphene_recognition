FROM node:18.17.0-alpine3.18

WORKDIR /app
COPY . .
RUN npm config set strict-ssl false
RUN  --mount=type=cache,target=/root/.npm npm install