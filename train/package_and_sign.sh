#!/bin/bash
# 一键打包模型并提交 sidecar 签名
# 用法: bash train/package_and_sign.sh <pkl_path> <step>
# 示例: bash train/package_and_sign.sh code/session_best/20260411-154102/best_model.pkl 9339

set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_DIR="$BASE/train"
CONTAINER="kaiwu-train-backup_model-1"
SIGN_DIR="$TRAIN_DIR/backup_model"   # bind mount: sidecar 签名后文件自动出现在此

# ---- 参数校验 ----
if [ $# -ne 2 ]; then
    echo "用法: bash train/package_and_sign.sh <pkl_path> <step>"
    echo "示例: bash train/package_and_sign.sh code/session_best/20260411-154102/best_model.pkl 9339"
    exit 1
fi

PKL_PATH="$1"
STEP="$2"

if [ ! -f "$PKL_PATH" ]; then
    echo "ERROR: 文件不存在: $PKL_PATH"
    exit 1
fi

if ! [[ "$STEP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: step 必须是数字: $STEP"
    exit 1
fi

# ---- 检查 sidecar 容器 ----
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: sidecar 容器 $CONTAINER 未运行"
    echo "请先启动训练: cd train && docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d"
    exit 1
fi

# ---- 打包 ----
echo "=== Step 1/3: 打包模型 ==="
python "$TRAIN_DIR/package_model.py" --pkl "$PKL_PATH" --step "$STEP"

# 找到最新生成的 zip（按修改时间排序取最新）
ZIP_FILE=$(ls -t "$TRAIN_DIR/_package_tmp/"*.zip 2>/dev/null | head -1)
if [ -z "$ZIP_FILE" ]; then
    echo "ERROR: 未找到生成的 zip 文件"
    exit 1
fi
ZIP_NAME=$(basename "$ZIP_FILE")
JSON_FILE="${ZIP_FILE}.json"

echo "  zip: $ZIP_FILE"
echo "  json: $JSON_FILE"

if [ ! -f "$JSON_FILE" ]; then
    echo "ERROR: 未找到 sidecar json: $JSON_FILE"
    exit 1
fi

# ---- 投递到 sidecar ----
echo ""
echo "=== Step 2/3: 投递到 sidecar 签名 ==="
docker cp "$ZIP_FILE" "$CONTAINER:/workspace/train/backup_model/"
docker cp "$JSON_FILE" "$CONTAINER:/workspace/train/backup_model/"
echo "  已投递: $ZIP_NAME"

# ---- 轮询等待签名完成 ----
echo ""
echo "=== Step 3/3: 等待签名完成 ==="
SIGNED_PATH="$SIGN_DIR/$ZIP_NAME"
TIMEOUT=60
ELAPSED=0

# 先删除宿主机上可能存在的旧同名签名文件（避免误判）
if [ -f "$SIGNED_PATH" ]; then
    rm -f "$SIGNED_PATH"
fi

while [ $ELAPSED -lt $TIMEOUT ]; do
    if [ -f "$SIGNED_PATH" ]; then
        SIGNED_SIZE=$(stat -c%s "$SIGNED_PATH" 2>/dev/null || stat -f%z "$SIGNED_PATH" 2>/dev/null)
        ORIGINAL_SIZE=$(stat -c%s "$ZIP_FILE" 2>/dev/null || stat -f%z "$ZIP_FILE" 2>/dev/null)
        # 签名后文件应大于原文件（.kaiwu.sign 被填充）
        if [ "$SIGNED_SIZE" -gt "$ORIGINAL_SIZE" ]; then
            echo ""
            echo "=== 签名完成 ==="
            echo "  签名文件: $SIGNED_PATH"
            echo "  文件大小: $(( SIGNED_SIZE / 1024 )) KB"
            echo ""
            echo "可直接提交此 zip 文件"
            exit 0
        fi
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    printf "\r  等待中... %ds/%ds" "$ELAPSED" "$TIMEOUT"
done

echo ""
echo "ERROR: 签名超时 (${TIMEOUT}s)"
echo "请检查 sidecar 日志: docker logs $CONTAINER"
exit 1
