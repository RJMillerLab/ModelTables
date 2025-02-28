#!/bin/bash

# 指定要检查的目录
DIRECTORY="data/downloaded_tex_files"

# 检查目录是否存在
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY does not exist."
    exit 1
fi

# 初始化有效和无效文件的数组
valid_files=()
invalid_files=()

# 遍历所有 .tar.gz 文件
for file in "$DIRECTORY"/*.tar.gz; do
    if [ -f "$file" ]; then
        echo "Checking $file..."
        # 尝试解压文件
        if tar -tzf "$file" > /dev/null 2>&1; then
            valid_files+=("$file")  # 添加到有效文件数组
            echo "$file is a valid tar.gz file."
        else
            invalid_files+=("$file")  # 添加到无效文件数组
            echo "$file is NOT a valid tar.gz file."
        fi
    else
        echo "No .tar.gz files found in $DIRECTORY."
    fi
done

# 输出有效和无效文件列表
echo "Valid .tar.gz files:"
printf '%s\n' "${valid_files[@]}"

echo "Invalid .tar.gz files:"
printf '%s\n' "${invalid_files[@]}"

