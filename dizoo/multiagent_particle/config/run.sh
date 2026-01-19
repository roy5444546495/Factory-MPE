#!/bin/bash

# 配置区域（只需修改这里）
SCRIPT_PATH="/root/marl_factory/dizoo/multiagent_particle/config/4vs4_alg1.py"

# 自动提取文件名（不含扩展名）
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
SCRIPT_NAME=$(basename "$SCRIPT_PATH" .py)
LOG_FILE="${SCRIPT_DIR}/${SCRIPT_NAME}.log"

# 执行
nohup python "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &
echo "Started: $SCRIPT_PATH"
echo "Log file: $LOG_FILE"
echo "PID: $!"