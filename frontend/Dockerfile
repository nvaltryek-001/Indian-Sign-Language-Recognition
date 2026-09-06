FROM nginx:1.27-alpine

ENV ISL_API_UPSTREAM=host.docker.internal:8000

COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY index.html styles.css app.js config.js action-mapping.json classes.json /usr/share/nginx/html/
COPY vendor /usr/share/nginx/html/vendor

EXPOSE 8080
