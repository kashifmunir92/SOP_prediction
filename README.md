
## Setup
### Install Dependencies

The code is based on huggaface's [transformers](https://github.com/huggingface/transformers). 

Install dependencies and [apex](https://github.com/NVIDIA/apex):
```
pip3 install -r requirement.txt
pip3 install --editable transformers
```


### Input data format

The input data format for our models is JSONL. Each line of the input file contains one document in the following format. To convert the labeled excel file placed under ```data/```, run ```to_json_data_v2.py 300```, ```300``` is the number of examples in dev dataset.
```
{"text":  "内蒙新华:新华集团始终担负着自治区蒙汉文中小学教材、大中专教材、幼教读物、政治读物及一般图书、音像制品、电子出版物的发行任务，还承担着全国八省区蒙文教材及蒙文一般图书的发行任务。",
"sentences": [["内", "蒙", "新", "华", ":", "新", "华", "集", "团", "始", "终", "担", "负", "着", "自", "治", "区", "蒙", "汉", "文", "中", "小", "学", "教", "材", "、", "大", "中", "专", "教", "材", "、", "幼", "教", "读", "物", "、", "政", "治", "读", "物", "及", "一", "般", "图", "书", "、", "音", "像", "制", "品", "、", "电", "子", "出", "版", "物", "的", "发", "行", "任", "务", "，", "还", "承", "担", "着", "全", "国", "八", "省", "区", "蒙", "文", "教", "材", "及", "蒙", "文", "一", "般", "图", "书", "的", "发", "行", "任", "务", "。"]],
# entities (boundaries and entity type)
"ner": [[[26, 30, "entity"], [52, 56, "entity"], [37, 40, "entity"], [17, 24, "entity"], [32, 35, "entity"], [0, 3, "entity"], [72, 75, "entity"], [42, 45, "entity"], [58, 59, "entity"], [47, 50, "entity"], [77, 82, "entity"]]],
# relations (two spans and relation type)
"relations": [[[0, 3, 17, 24, "主营"], [0, 3, 26, 30, "主营"], [0, 3, 32, 35, "主营"], [0, 3, 37, 40, "主营"], [0, 3, 42, 45, "主营"], [0, 3, 47, 50, "主营"], [0, 3, 52, 56, "主营"], [0, 3, 72, 75, "主营"], [0, 3, 77, 82, "主营"], [58, 59, 17, 24, "domain_relation"], [58, 59, 26, 30, "domain_relation"], [58, 59, 32, 35, "domain_relation"], [58, 59, 37, 40, "domain_relation"], [58, 59, 42, 45, "domain_relation"], [58, 59, 47, 50, "domain_relation"], [58, 59, 52, 56, "domain_relation"], [58, 59, 72, 75, "domain_relation"], [58, 59, 77, 82, "domain_relation"]]]
}
```

### Trained Models
The model can be initialized with Pretrained Chinese BERT (Pytorch). 
I further pretrained the PLM on in-domain dataset to find-tune the language model. 



## Training Script

Train and evaluate NER Model: (```max_mention_ori_length``` defines the maximum span length)
```
CUDA_VISIBLE_DEVICES=0  python3  run_acener.py  --model_type bertspanmarker  \
    --model_name_or_path  bert_model/  --do_lower_case  \
    --data_dir data/json/  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 1400  --max_pair_length 256  --max_mention_ori_length 20    \
     --do_train --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
  --seed 42  --onedropout  --lminit  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_dir output_results/PL-Marker-42  --overwrite_output_dir  --output_results
```

Copy the NER predicted results to ```data/json/``` by running ```cp output_results/PL-Marker-42/ent_pred_test.json data/json```


Train and evaluate RE model:
```
CUDA_VISIBLE_DEVICES=0  python3  run_re.py  --model_type bertsub  \
    --model_name_or_path  bert_model/  --do_lower_case  \
    --data_dir data/json/  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  16  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --max_pair_length 40  --save_steps 1000  \
     --do_train --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
     --seed 42  \
    --train_file train.json  --dev_file dev.json  \
    --test_file ent_pred_test.json  \
    --output_dir output_results/re-bert-42  --overwrite_output_dir
```
Here,  `--use_ner_results` denotes using the original entity type predicted by NER model.


