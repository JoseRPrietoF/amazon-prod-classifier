#!/usr/bin/env bash
host=$1
shift
port=$1
shift

while ! nc -z "$host" "$port"; do
  echo "Waiting for $host:$port..."
  sleep 1
done

exec "$@"
