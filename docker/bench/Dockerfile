FROM python:3.12.10-slim-bookworm
WORKDIR /bench
COPY docker/bench/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY bench/ /bench/
COPY data/ /bench/data/
COPY scripts/run_all.sh /bench/run_all.sh
ENV PYTHONUNBUFFERED=1 RESULTS_DIR=/results
CMD ["sh", "/bench/run_all.sh"]
