
# ASCEND_RT_VISIBLE_DEVICES=0 python test_qlora.py --train_args_file train_args/qlora/llama2-sft-qlora.json \
# &> logs/llama2-test.log

ASCEND_RT_VISIBLE_DEVICES=5 python test_qlora.py --train_args_file train_args/qlora/Qwen2.5-sft-qlora.json \
&> logs/Qwen2.5-test.log

