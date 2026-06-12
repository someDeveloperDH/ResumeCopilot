CPU용 Stage 3 파일입니다.

적용 위치:
  ResumeCopilot/stage3_multitask/

사용법:
  1) 기존 train.py / evaluate.py / config.yaml 백업
  2) 이 폴더의 3개 파일을 stage3_multitask에 덮어쓰기
  3) pip install torch transformers pandas pyyaml scikit-learn scipy
  4) python train.py
  5) python evaluate.py

변경점:
  - answerdotai/ModernBERT-base → monologg/koelectra-small-v3-discriminator
  - max_seq_length 512 → 256
  - batch_size 8 → 4
  - CPU 강제 사용
