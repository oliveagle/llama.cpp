#!/bin/bash

# llama.cpp CPU vs GPU 推理性能对比演示 (使用 llama-server)
# 用于 Bonsai-8B 模型在 Mac M4 上的性能测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="/Users/oliveagle/.lmstudio/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B.gguf"
LLAMA_SERVER="./build-metal/bin/llama-server"
PORT_CPU=8080
PORT_GPU=8081
HOST="localhost"
PROMPT="2+2="
TEMP=0.5
N_PREDICT=15

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     llama.cpp CPU vs GPU 推理性能对比 (llama-server API)      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📦 模型: Bonsai-8B.gguf (Q1_0_g128, ~1.1 GB)"
echo "🖥️  设备: Mac M4"
echo "🌡️  温度: $TEMP"
echo "📝 提示: $PROMPT"
echo ""

# 检查文件
if [ ! -f "$MODEL" ]; then
    echo -e "${RED}❌ 错误: 模型文件不存在: $MODEL${NC}"
    exit 1
fi

if [ ! -f "$LLAMA_SERVER" ]; then
    echo -e "${RED}❌ 错误: llama-server 不存在: $LLAMA_SERVER${NC}"
    exit 1
fi

# 获取毫秒级时间戳 (macOS 兼容)
get_time_ms() {
    python3 -c "import time; print(int(time.time() * 1000))"
}

# 测试推理性能
test_inference() {
    local backend_name=$1
    local gpu_layers=$2
    local port=$3

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}📊 测试: $backend_name 推理 (--gpu-layers $gpu_layers)${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    local log_file="/tmp/llama_server_${backend_name// /_}_$$.log"

    # 启动服务器
    echo -e "${YELLOW}🚀 启动 llama-server...${NC}"
    $LLAMA_SERVER \
        --model "$MODEL" \
        --host $HOST \
        --port $port \
        --gpu-layers $gpu_layers \
        --temp $TEMP \
        > "$log_file" 2>&1 &

    local server_pid=$!

    # 等待模型完全加载
    echo -e "${YELLOW}⏳ 等待模型加载...${NC}"
    local max_wait=120
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if grep -q "model loaded" "$log_file" 2>/dev/null; then
            echo -e "${GREEN}✅ 模型已加载${NC}"
            break
        fi
        if ! kill -0 $server_pid 2>/dev/null; then
            echo -e "${RED}❌ 服务器异常退出${NC}"
            cat "$log_file" | tail -30
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
        echo -n "."
    done

    if [ $waited -ge $max_wait ]; then
        echo -e "${RED}❌ 模型加载超时${NC}"
        cat "$log_file" | tail -30
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    echo ""

    # 额外等待确保完全就绪
    sleep 2

    # 发送推理请求
    echo -e "${YELLOW}⚙️  发送推理请求...${NC}"
    local start_time=$(get_time_ms)

    local response=$(curl -s -X POST "http://$HOST:$port/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$PROMPT\",
            \"temperature\": $TEMP,
            \"n_predict\": $N_PREDICT
        }")

    local end_time=$(get_time_ms)
    local elapsed=$((end_time - start_time))

    # 提取生成内容
    local result=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('content', '').strip())" 2>/dev/null || echo "")

    # 停止服务器
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true

    # 等待端口释放
    sleep 2

    # 显示结果
    echo ""
    echo -e "${GREEN}✨ 生成结果: $result${NC}"
    echo -e "${GREEN}⏱️  总耗时: ${elapsed} ms${NC}"

    # 保存结果
    if [ "$backend_name" == "CPU" ]; then
        CPU_RESULT="$result"
        CPU_TIME="$elapsed"
    else
        GPU_RESULT="$result"
        GPU_TIME="$elapsed"
    fi
}

# 清理可能存在的旧进程
ps aux | awk '/[l]lama-server/{print $2}' | xargs kill 2>/dev/null || true
sleep 2

# 运行测试
test_inference "CPU" 0 $PORT_CPU
test_inference "GPU (Metal)" 99 $PORT_GPU

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 性能对比总结"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "%-20s %-20s %-30s\n" "后端" "耗时 (ms)" "结果"
printf "%-20s %-20s %-30s\n" "--------------------" "--------------------" "------------------------------"
printf "%-20s %-20s %-30s\n" "CPU" "${CPU_TIME:-N/A}" "${CPU_RESULT:-N/A}"
printf "%-20s %-20s %-30s\n" "GPU (Metal)" "${GPU_TIME:-N/A}" "${GPU_RESULT:-N/A}"
echo ""

# 计算加速比
if [ -n "$CPU_TIME" ] && [ -n "$GPU_TIME" ] && [ "$CPU_TIME" -gt 0 ] && [ "$GPU_TIME" -gt 0 ]; then
    speedup=$(python3 -c "print(f'{float($CPU_TIME) / float($GPU_TIME):.2f}')")
    echo -e "${GREEN}🚀 GPU 加速比: ${speedup}x${NC}"
    echo ""
fi

echo "📝 结论:"
echo "  • llama-server API 正常工作"
echo "  • CPU 和 Metal 后端都已启用"
echo "  • 可通过 HTTP API 进行推理"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
