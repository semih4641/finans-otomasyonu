#!/bin/sh
set -eu

data_dir="${BOT_DATA_DIR:-/data}"

mkdir -p "$data_dir/backtest_out" "$data_dir/models/bist_v2"

if [ ! -s "$data_dir/signal_state.json" ]; then
  printf '{}\n' > "$data_dir/signal_state.json"
fi

if [ ! -f "$data_dir/models/bist_v2/active.json" ] && [ -d /opt/model-seed/bist_v2 ]; then
  cp -a /opt/model-seed/bist_v2/. "$data_dir/models/bist_v2/"
fi

touch "$data_dir/finans_bot.log"

exec "$@"
