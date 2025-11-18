mkdir benchmark/data
mkdir benchmark/data/infinite-bench
mkdir benchmark/data/longbench

python benchmark/download.py

curl https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/kv_retrieval.jsonl
