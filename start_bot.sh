#!/usr/bin/env bash
# Fill your secrets in .env (copy .env.example -> .env)
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
python main.py
