Training:

CUDA_VISIBLE_DEVICES=0 python -u train.py  --model_name_or_path t5-base  --train_file ../dataset/train/questions_t5.json --validation_file ../dataset/valid/questions_t5.json     --source_prefix "qa: "     --output_dir model1    --per_device_train_batch_size=16 --per_device_eval_batch_size=100 --learning_rate=1e-4 --num_train_epochs=100   | tee logs/log1.txt

Inference:

python infer_api.py  --model_name_or_path model1/epoch_5/



Questions:

curl --header "Content-Type: application/json"  --request POST  --data '{"question":"How many papers has Chris Biemann authored ?"}'  http://localhost:5000/answer


