FROM nginx:alpine

# Build context is the project root so we can grab static/ from there.
COPY static/ /usr/share/nginx/html/
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
