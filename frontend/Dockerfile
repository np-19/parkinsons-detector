# ==============================================================================
# Production Dockerfile for NeuroVocal React / TypeScript Frontend
# Multi-stage build producing lightweight Nginx container (~25 MB)
# ==============================================================================

FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]

