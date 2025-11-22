config=config/mistral-inf-llm.yaml

mkdir benchmark/infinite-bench-result

python benchmark/pred.py \
--config_path ${config} \
--output_dir_path benchmark/infinite-bench-result \
--datasets kv_retrieval

python benchmark/eval.py --dir_path benchmark/infinite-bench-result