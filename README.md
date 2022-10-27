PRINCE: Prefix-Masked Decoding for Knowledge Enhanced Sequence-to-Sequence Pre-Training
===



## Introduction

PRINCE: Prefix-Masked Decoding for Knowledge Enhanced Sequence-to-Sequence Pre-Training. EMNLP 2022.

## Dependencies


Currently we implement PRINCE based on Fairseq. The depencies are as follows:

- Python version >= 3.6
- PyTorch version >= 1.5.0
- Fairseq version >= 0.10.2



## Pre-training:


we pretrain our model on wikidata dataset, which can be download from [KGPT](https://github.com/wenhuchen/KGPT), 

<!--same data of bart.-->

#### Preprocess


```
# 1. BPE

# 2. Split parts, valid, test
python split.py
```

Binarize dataset:
```
# For Training data
DATA_DIR=/workspace/fairseq-models/data/mono/hrli
for bpe_file in ${DATA_DIR}/bpe/parts/train.*.bpe;
do
  echo "Start" $bpe_file "@" `date`
  idx=$(echo $(basename ${bpe_file}) | cut -f2 -d '.')
  BIN_DIR=${DATA_DIR}/bin/part${idx}
  ## binary txt
  fairseq-preprocess \
    --only-source \
    --srcdict ${DATA_DIR}/bpe/dict.txt \
    --trainpref  ${bpe_file} \
    --destdir  ${BIN_DIR} \
    --workers 100 > /dev/null
  echo "End  " $bpe_file  "@" `date`
done

# For valid data
fairseq-preprocess \
  --only-source \
  --srcdict ${DATA_DIR}/bpe/dict.txt \
  --validpref  ${DATA_DIR}/bpe/valid.bpe \
  --destdir  ${DATA_DIR}/bin/valid \
  --workers 100 > /dev/null

for part in part*;
do
  cp valid/valid* $part
done  
```

#### Pre-train

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3

function join_by { local IFS="$1"; shift; echo "$*"; }
DATA_DIR=$(join_by : data/mono/hrli/bin/part*)  # 

# RESTORE_MODEL=models/fairseq/bart.base/model.pt   # base model
# ARCH=transformer_dtf_base  # base model 

RESTORE_MODEL=models/fairseq/bart.large/model.pt    # large mode 
ARCH=transformer_dtf_large  # large model

# mask_param=""  # no decoder mask
mask_param="--apply-decoder-mask --decoder-mask-prob 0.25"   # decoder mask in random position
# mask_param="--apply-decoder-mask --only-mask-entity-in-decoder"  # decoder mask in entity position

fairseq-train \
  ${DATA_DIR} \
  --user-dir src \
  --task denosing_tf --arch $ARCH \
  --criterion auto_detect \
  ${mask_param} \
  --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed \
  --reset-optimizer --reset-dataloader --reset-meters \
  --apply-bert-init --max-source-positions 1024 --max-target-positions 1024 \
  --skip-invalid-size-inputs-valid-test --optimizer adam --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-6 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 3e-05 \
  --warmup-updates 500 --total-num-update 125000 --dropout 0.1 --attention-dropout 0.1 \
  --weight-decay 0.01 --max-tokens 2048 \
  --ddp-backend=no_c10d  \
  --restore-file $RESTORE_MODEL
```


