#!/bin/bash
# GGUF Model Usage Script
# Use this script in environments with better llama.cpp support

# Download the model (if not already local)
# wget https://huggingface.co/coderop12/gemma2b-nirf-lookup-2025-gguf-v2/resolve/main/gemma2b-nirf-lookup-2025-q4_k_m.gguf

# Option 1: Direct llama.cpp usage
./llama.cpp/build/bin/llama-cli \
    --model gemma2b-nirf-lookup-2025-q4_k_m.gguf \
    --prompt "What is the ranking of IIT Madras in NIRF 2025?" \
    --n-predict 100 \
    --ctx-size 2048 \
    --temp 0.7

# Option 2: Interactive mode
./llama.cpp/build/bin/llama-cli \
    --model gemma2b-nirf-lookup-2025-q4_k_m.gguf \
    --interactive

# Option 3: Python with llama-cpp-python (when working)
# python3 -c "
# from llama_cpp import Llama
# llm = Llama('gemma2b-nirf-lookup-2025-q4_k_m.gguf')
# print(llm('What is IIT Madras ranking?', max_tokens=100))
# "
