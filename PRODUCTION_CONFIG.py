"""
CMFO Mining Intelligence - Production Configuration
===================================================

Maximum performance settings for production deployment.
"""

# GPU Configuration
GPU_CONFIG = {
    'num_gpus': 4,  # Use all available GPUs
    'threads_per_gpu': 262144,  # Maximum threads
    'blocks_per_gpu': 1024,
    'threads_per_block': 256,
    'enable_tensor_cores': True,
    'mixed_precision': True  # FP16 for speed
}

# AI Model Configuration
AI_CONFIG = {
    'batch_size': 256,  # Large batches for GPU efficiency
    'num_workers': 16,  # Parallel data loading
    'pin_memory': True,
    'device': 'cuda',
    'compile_model': True,  # PyTorch 2.0 compilation
    'inference_mode': True  # Disable gradient tracking
}

# Memory Configuration
MEMORY_CONFIG = {
    'historical_blocks': 100000,  # Full history
    'mempool_buffer': 50000,
    'template_cache': 1000,
    'enable_pruning': True
}

# Mining Configuration
MINING_CONFIG = {
    'templates_per_cycle': 100,
    'max_search_time': 10.0,  # seconds
    'pruning_threshold': 0.99,  # Aggressive pruning
    'phase_tolerance': 0.05,
    'enable_adaptive_difficulty': True
}

# Network Configuration
NETWORK_CONFIG = {
    'bitcoin_rpc_url': 'http://localhost:8332',
    'rpc_user': 'bitcoin',
    'rpc_password': 'password',
    'rpc_timeout': 30,
    'mempool_refresh_rate': 0.1  # 100ms
}

# Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'enable_metrics': True,
    'metrics_interval': 60,  # seconds
    'log_file': 'cmfo_mining.log'
}
