# PostgreSQL 17 + pgContext 0.3.0 (Evokoa release image) + pgvector 0.8.6 (official image), release binaries unmodified.
# Both images are PostgreSQL 17 on Debian; vector.so is copied into the pgContext image, as in the original experiment.
ARG PGCONTEXT_IMAGE=ghcr.io/evokoa/pgcontext:pg17-v0.3.0
ARG PGVECTOR_IMAGE=pgvector/pgvector:0.8.6-pg17

FROM ${PGVECTOR_IMAGE} AS pgvector

FROM ${PGCONTEXT_IMAGE}
LABEL org.opencontainers.image.title="pgctx-repro-pg" \
      org.opencontainers.image.description="PostgreSQL 17 + pgContext 0.3.0 + pgvector 0.8.6 (reproduction of a pgContext/pgvector comparison)"
COPY --from=pgvector /usr/lib/postgresql/17/lib/vector.so /usr/lib/postgresql/17/lib/
COPY --from=pgvector /usr/share/postgresql/17/extension/vector* /usr/share/postgresql/17/extension/
RUN rm -f /docker-entrypoint-initdb.d/*
COPY init/ /docker-entrypoint-initdb.d/
