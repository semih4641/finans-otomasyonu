FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    TZ=Europe/Istanbul

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p /opt/model-seed \
    && cp -a /app/models/bist_v2 /opt/model-seed/bist_v2 \
    && rm -rf /app/backtest_out /app/models/bist_v2 /app/signal_state.json /app/finans_bot.log \
    && ln -s /data/backtest_out /app/backtest_out \
    && ln -s /data/models/bist_v2 /app/models/bist_v2 \
    && ln -s /data/signal_state.json /app/signal_state.json \
    && ln -s /data/finans_bot.log /app/finans_bot.log \
    && chmod +x /app/deploy/docker-entrypoint.sh \
    && chown -R 1000:1000 /app /opt/model-seed

USER 1000:1000

ENTRYPOINT ["/app/deploy/docker-entrypoint.sh"]
CMD ["python", "bot.py"]
