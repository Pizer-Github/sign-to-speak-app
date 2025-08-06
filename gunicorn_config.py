import multiprocessing
import os

# Gunicorn configuration for Render deployment
# Optimized for memory-constrained environments

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1  # Single worker to save memory
worker_class = "sync"
worker_connections = 100
timeout = 120  # Increased timeout for heavy processing
keepalive = 2
max_requests = 100  # Restart worker after processing requests
max_requests_jitter = 10
preload_app = True  # Load app before forking workers
worker_tmp_dir = "/dev/shm"  # Use memory for temp files if available

# Memory management
worker_memory_limit = 400 * 1024 * 1024  # 400MB limit per worker
worker_rlimit_as = 500 * 1024 * 1024     # Virtual memory limit

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Graceful shutdown
graceful_timeout = 30
worker_tmp_dir = "/dev/shm"