###################
# Train Baseline
###################

dataset='Visual_YOLO_CLIP'
SEED=2024
CUDA_DEVICE=7


# 用来训练 nlpcc18训练集的
MODEL_DIR_STAGE1=../../gopar/chinese_${dataset}_bart_baseline/
PROCESSED_DIR_STAGE1=../../preprocess_data/chinese_${dataset}_with_syntax_bart
BASE_PATH=../../../../../../model/SynGEC/src/src_syngec/

FAIRSEQ_PATH=${BASE_PATH}/fairseq2
FAIRSEQ_CLI_PATH=${FAIRSEQ_PATH}/fairseq_cli
SYNGEC_MODEL_PATH=${BASE_PATH}/syngec_model


mkdir -p $MODEL_DIR_STAGE1
mkdir -p $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE1/src

cp -r $SYNGEC_MODEL_PATH $MODEL_DIR_STAGE1/src

cp ./2_train_syngec_bart.sh $MODEL_DIR_STAGE1

# Transformer-base-setting stage 1
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE1/bin \
    --save-dir $MODEL_DIR_STAGE1 \
    --user-dir $SYNGEC_MODEL_PATH \
    --bart-model-file-from-transformers fnlp/bart-large-chinese \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_bart_large \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 2048 \
    --optimizer adam \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --lr 3e-05 \
    --warmup-updates 2000 \
    -s src \
    -t tgt \
    --lr-scheduler polynomial_decay \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --share-all-embeddings \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --keep-last-epochs 10 \
    --patience 5 \
    --seed $SEED 2>&1 | tee ${MODEL_DIR_STAGE1}/nohup.log
wait

#####################################################################
# Train SynGEC
#####################################################################

#SEED=2023
#dataset='Visual_OCR'
#
#
## 训练nlpcc18的训练集
#MODEL_DIR_STAGE1=../../gopar/chinese_${dataset}_bart_syngec/
#PROCESSED_DIR_STAGE1=../../preprocess_data/chinese_${dataset}_with_syntax_bart
#BASE_PATH=../../../../../../model/SynGEC/src/src_syngec/
#
#FAIRSEQ_PATH=${BASE_PATH}/fairseq2
#FAIRSEQ_CLI_PATH=${FAIRSEQ_PATH}/fairseq_cli
#SYNGEC_MODEL_PATH=${BASE_PATH}/syngec_model
#
## 训练nlpcc18的训练集
#BART_PATH=../../gopar/chinese_${dataset}_bart_baseline/checkpoint_best.pt
#
#mkdir -p $MODEL_DIR_STAGE1
#mkdir -p $MODEL_DIR_STAGE1/src
#cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE1/src
#cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE1/src
#cp -r $SYNGEC_MODEL_PATH $MODEL_DIR_STAGE1/src
#cp ./2_train_syngec_bart.sh $MODEL_DIR_STAGE1
#
#CUDA_VISIBLE_DEVICES=7 python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE1/bin \
#    --save-dir $MODEL_DIR_STAGE1 \
#    --user-dir $SYNGEC_MODEL_PATH \
#    --use-syntax \
#    --only-gnn \
#    --syntax-encoder GCN \
#    --freeze-bart-parameters \
#    --finetune-from-model $BART_PATH \
#    --task syntax-enhanced-translation \
#    --arch syntax_enhanced_bart_large \
#    --skip-invalid-size-inputs-valid-test \
#    --max-tokens 2048 \
#    --optimizer adam \
#    --max-source-positions 512 \
#    --max-target-positions 512 \
#    --max-sentence-length 128 \
#    --lr 5e-04 \
#    --warmup-updates 2000 \
#    -s src \
#    -t tgt \
#    --lr-scheduler polynomial_decay \
#    --clip-norm 1.0 \
#    --criterion label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --max-epoch 60 \
#    --share-all-embeddings \
#    --adam-betas '(0.9,0.999)' \
#    --log-format tqdm \
#    --find-unused-parameters \
#    --fp16 \
#    --keep-last-epochs 10 \
#    --patience 5 \
#   --seed $SEED 2>&1 | tee ${MODEL_DIR_STAGE1}/nohup.log
#wait
