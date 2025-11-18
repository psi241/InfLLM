config=config/mistral-inf-llm.yaml

datasets="narrativeqa"

mkdir benchmark/longbench-result

python benchmark/pred.py \
--config_path ${config} \
--output_dir_path benchmark/longbench-result \
--datasets ${datasets} 

python benchmark/eval.py --dir_path benchmark/longbench-result