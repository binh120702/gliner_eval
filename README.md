
# Chuyên đề nghiên cứu về một số vấn đề chọn lọc trong khoa học máy tính - CS2311.CH190

GLiNER is a generalist model for NER (Extract any entity types from texts)

- [Source code](https://github.com/urchade/GLiNER)
- [Paper](https://aclanthology.org/2024.naacl-long.300/)

Bài báo cáo này nhằm re-produce lại kết quả bài báo và chạy thử nghiệm trên dataset VLSP 2018.  

# Chi tiết thực nghiệm 
**(File log training và kết quả evaluation được lưu trong folder code - logs_and_results/)**

## Danh sách các model nhóm đã tái thực nghiệm:

- GLiNER S: model cỡ nhỏ (Deberta-v3-small) [link](https://huggingface.co/binhpdt/reproduced-gliner-small)
  - train_log: [train_small.log](logs_and_results/train_small.log)
- GLiNER M: model cỡ trung (Deberta-v3-base) [link](https://huggingface.co/binhpdt/reproduced-gliner-medium)
  - train_log: [train_medium.log](logs_and_results/train_medium.log)
- GLiNER L: model cỡ lớn (Deberta-v3-large) [link](https://huggingface.co/binhpdt/reproduced-gliner-large) (Train với batch size 8, steps 30000) tuy nhiên do thiếu hụt tài nguyên GPU nên sẽ có hiện tượng skip steps khi training. Do đó -> GLiNER L-mod
  - train_log: [train_large.log](logs_and_results/train_large.log)
- GLiNER L-mod: model cỡ lớn (mod)(Deberta-v3-large) [link](https://huggingface.co/binhpdt/reproduced-gliner-large) (Train với batch size 4, steps 30000) Train nhiều hơn 30000 steps sẽ không cải thiện kết quả quá nhiều.
  - train_log: [train_large_loss.log](logs_and_results/train_large_loss.log)
- GLiNER Multi: model đa ngôn ngữ (mDeberta-v3-base) [link](https://huggingface.co/binhpdt/reproduced-gliner-multi)
  - train_log: [train_multi.log](logs_and_results/train_multi.log)
- GLiNER Bert: train trên backbone BERT-base-uncased [link](https://huggingface.co/binhpdt/reproduced-bert-medium)
  - train_log: [train_bert.log](logs_and_results/train_bert.log)
- GLiNER Albert: train trên backbone Albert-base-v2 [link](https://huggingface.co/binhpdt/reproduced-albert-medium)
  - train_log: [train_albert.log](logs_and_results/train_albert.log)
- GLiNER Roberta: train trên backbone RoBERTa-base [link](https://huggingface.co/binhpdt/reproduced-roberta-medium)
  - train_log: [train_roberta.log](logs_and_results/train_roberta.log)
- GLiNER Electra: train trên backbone Electra-base-generator [link](https://huggingface.co/binhpdt/reproduced-electra-medium)
  - train_log: [train_electra.log](logs_and_results/train_electra.log)
- GLiNER mix20 train: train từ đầu trên tập 20NER [link](https://huggingface.co/binhpdt/reproduced-20ner-mixed-train-gliner)
  - train_log: [train_mix_train.log](logs_and_results/train_mix_train.log)
- GLiNER mix20 finetune: train trên tập pileNER sau đó finetune trên tập 20NER [link](https://huggingface.co/binhpdt/reproduced-20ner-mixed-tune-gliner)
  - train_log: [train_mix_tune.log](logs_and_results/train_mix_tune.log)
- GLiNER VLSP18: train trên tập VLSP 2018 [link](https://huggingface.co/binhpdt/gliner-vlsp18)
  - train_log: [train_vlsp.log](logs_and_results/train_vlsp.log)

## Danh sách bộ dữ liệu thực nghiệm:
- PileNER: Bộ dữ liệu gốc [link_huggingface](https://huggingface.co/datasets/Universal-NER/Pile-NER-type/resolve/main/train.json)
- 20NER và OOD NER: tuyển tập các bộ dữ liệu đánh giá cho NER [link_drive](https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view)
- MultiCONER: bộ dữ liệu đa ngôn ngữ [link_kaggle](https://www.kaggle.com/datasets/davindersingh23031/multiconer/data) or [offical](https://registry.opendata.aws/multiconer/)
- VLSP 2018-NER (vui lòng liên hệ VLSP để access)
## Bảng 1 và 2: 
- Bao gồm chạy thực nghiệm 3 model S, M, L trên tập 20NER và tập OOD NER:
- eval_log:
  - [eval_small.log](logs_and_results/eval_small.log)
  - [eval_medium.log](logs_and_results/eval_medium.log)
  - [eval_large_loss.log](logs_and_results/eval_large_loss.log)
- eval_result:
  - [logs_large_med_small/results.txt](logs_and_results/logs_large_med_small/results.txt)
## Bảng 3:
- Bao gồm thực nghiệm model large và model multi trên bộ dataset MultiCONER:
- Hàm tiền xử lí dữ liệu được nhóm triển khai trong file [gliner/evaluation/evaluate_multiconer.py](gliner/evaluation/evaluate_multiconer.py)
- eval_results:
  - Model large: [logs_multi_large/results.txt](logs_and_results/logs_multi_large/results.txt)
  - Model multi: [logs_multi_multi/results.txt](logs_and_results/logs_multi_multi/results.txt)
  
## Bảng 4:
- Bao gồm thực nghiệm model VLSP 2018 với model large finetune VLSP, large không finetune VLSP và model multi:
- Hàm tiền xử lí dữ liệu được nhóm triển khai trong file [gliner/evaluation/evaluate_vlsp.py](gliner/evaluation/evaluate_vlsp.py)
- eval_results:
  - All in one: [logs_vlsp_4k/results.txt](logs_and_results/logs_vlsp_4k/results.txt)
## Figure 5:
- Bao gồm thực nghiệm so sánh các backbone khác nhau:
- eval_results:
  - Bert: [logs_bert/results.txt](logs_and_results/logs_bert/results.txt)
  - Albert: [logs_albert/results.txt](logs_and_results/logs_albert/results.txt)
  - Roberta: [logs_roberta/results.txt](logs_and_results/logs_roberta/results.txt)
  - Electra: [logs_electra/results.txt](logs_and_results/logs_electra/results.txt)
  - Deberta-v3 (chính là GLiNER medium) [log_gliner_medium/results.txt](logs_and_results/logs_gliner_medium/results.txt)
## Bảng 5:
- Bao gồm việc sample 10k trong mỗi bộ dataset thuộc bộ 20NER, tạo thành 1 bộ mix_data và cho finetune trên GLiNER để đánh giá.
- eval_results:
  - Model train từ đầu: [logs_mix_train/results.txt](logs_and_results/logs_mix_train/results.txt)
  - Model train trước trên PileNER sau đó finetune trên mix_data: [logs_mix_tune/results.txt](logs_and_results/logs_mix_tune/results.txt)

# Novelty

- Triển khai hàm loss adversarial contrastive loss, span embedding và entity embedding sai sẽ được đẩy xa khỏi nhau, đặc biệt là cho các cặp entity "gần nhau / lẫn lộn"
  - Kết hợp với focal loss:
    - Giữ nguyên focal loss
    - Thêm thành phần contrastive với trọng số cấu hình được
    - Cân bằng học phân loại và học embedding
  - Train 3 mô hình small, medium, large với hàm loss này và so sánh kết quả với mô hình original.
  - gliner_new_loss_small: 
    - train_log: [logs/results_small_loss.txt](logs/train_small_loss.txt)
    - link model huggingface: [gliner_new_loss_small](https://huggingface.co/binhpdt/gliner_constractive_loss_small)
    - eval_results: [logs/results_small_loss.txt](logs/results_small_loss.txt)
  - gliner_new_loss_medium: 
    - train_log: [logs/results_medium_loss.txt](logs/train_medium_loss.txt)
    - link model huggingface: [gliner_new_loss_medium](https://huggingface.co/binhpdt/gliner_constractive_loss_medium)
    - eval_results: [logs/results_medium_loss.txt](logs/results_medium_loss.txt)
  - gliner_new_loss_large: 
    - train_log: [logs/results_large_loss.txt](logs/train_large_loss.txt)
    - link model huggingface: [gliner_new_loss_large](https://huggingface.co/binhpdt/gliner_constractive_loss_large)
    - eval_results: [logs/results_large_loss.txt](logs/results_large_loss.txt)
- Giới thiệu phương pháp decode span: greedy(default), nms, mwis. Chạy thử trên mô hình `gliner_new_loss_large`
  - eval_results_mwis: [logs/test_eval_mwis/results.txt](logs/test_eval_mwis/tables.txt)
  - eval_results_nms: [logs/test_eval_nms/results.txt](logs/test_eval_nms/tables.txt)