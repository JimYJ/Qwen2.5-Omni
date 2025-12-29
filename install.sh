#!/usr/bin/env bash

# 所有输出重定向到 install.log，并在后台执行
{
  python -m pip install transformers==4.52.3
  python -m pip install accelerate
  python -m pip install gptqmodel==2.0.0
  python -m pip install numpy==2.0.0
} > install.log 2>&1 &