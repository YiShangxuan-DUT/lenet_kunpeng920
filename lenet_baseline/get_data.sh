#!/usr/bin/env bash
set -euo pipefail

echo "▶ 准备目录 …"
mkdir -p data weight

# ====== MNIST 下载 ======
echo "▶ 下载 MNIST 数据集到 ./data"

# 3 个 HTTPS 镜像（依次尝试）
mirrors=(
  "https://ossci-datasets.s3.amazonaws.com/mnist"
  "https://storage.googleapis.com/cvdf-datasets/mnist"
  "http://yann.lecun.com/exdb/mnist"          # 最后再试原站
)

files=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

download() {            # $1 = 文件名
  local f="$1"
  for m in "${mirrors[@]}"; do
    url="$m/$f"
    echo "  → $url"
    if curl -L -o "data/$f" "$url" \
        --retry 4 --retry-connrefused --connect-timeout 10 --progress-bar; then
      return 0
    else
      echo "    × 失败，换下一个源 …"
    fi
  done
  echo "‼ 所有镜像都失败：$f" && return 1
}

for f in "${files[@]}"; do
  if [[ -f "data/$f" ]]; then
    echo "✔ $f 已存在，跳过"
  else
    download "$f"
  fi
done

echo "▶ 解压 MNIST 数据集到 ./data"
for f in data/*.gz; do
  echo "  → 解压 $f"
  gzip -d "$f"
done

# ====== 权重下载 ======
echo "▶ 下载预训练 LeNet-5 权重到 ./weight"
curl -L -o weight/lenet5_std.pth \
     "https://gh-proxy.com/raw.githubusercontent.com/YiShangxuan-DUT/lenet_arm64/main/lenet5_std.pth" \
     --retry 4 --retry-connrefused --connect-timeout 10 --progress-bar

echo -e "\n✓ 全部完成！\n   数据集：$(pwd)/data\n   权  重：$(pwd)/weight"

