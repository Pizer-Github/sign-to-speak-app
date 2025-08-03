#!/bin/bash
# Start script for Render deployment
gunicorn --bind 0.0.0.0:$PORT app:app
