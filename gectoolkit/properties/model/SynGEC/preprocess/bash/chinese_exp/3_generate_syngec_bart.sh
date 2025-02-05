CUDA_DEVICE=7
BEAM=12
N_BEST=1
dataset='Visual_OCR'


BASE_PATH=../../../../../../model/SynGEC/src/src_syngec
FAIRSEQ_PATH=${BASE_PATH}/fairseq2
FAIRSEQ_DIR=${FAIRSEQ_PATH}/fairseq_cli
SYNGEC_MODEL_PATH=${BASE_PATH}/syngec_model


# train,valid数据集
PROCESSED_DIR=../../preprocess_data/chinese_${dataset}_with_syntax_bart
TEST_DIR=../../preprocess_data/chinese_${dataset}_test_with_syntax_bart
TEST_PROCESSED_DIR=${TEST_DIR}/bin

MODEL_DIR=../../gopar/chinese_${dataset}_bart_syngec
OUTPUT_DIR=${MODEL_DIR}/results

mkdir -p $OUTPUT_DIR
#cp $ID_FILE $OUTPUT_DIR/handwritten.id
cp $TEST_DIR/test.char.src $OUTPUT_DIR/handwritten.src.char

echo "Generating handwritten Test..."
SECONDS=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u ${FAIRSEQ_DIR}/interactive.py $PROCESSED_DIR/bin \
    --user-dir ${SYNGEC_MODEL_PATH} \
    --task syntax-enhanced-translation \
    --path ${MODEL_DIR}/checkpoint_best.pt \
    --beam ${BEAM} \
    --nbest ${N_BEST} \
    -s src \
    -t tgt \
    --buffer-size 10000 \
    --batch-size 32 \
    --num-workers 12 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    --conll_file ${TEST_PROCESSED_DIR}/test.conll.src-tgt.src \
    --dpd_file ${TEST_PROCESSED_DIR}/test.dpd.src-tgt.src \
    --probs_file ${TEST_PROCESSED_DIR}/test.probs.src-tgt.src \
    --output_file ${OUTPUT_DIR}/${dataset}.out.nbest \
    < ${OUTPUT_DIR}/${dataset}.src.char

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

cat ${OUTPUT_DIR}/${dataset}.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/${dataset}.out
sed -i '$d' $OUTPUT_DIR/${dataset}.out
python ../../utils/post_process_chinese.py $OUTPUT_DIR/${dataset}.src.char $OUTPUT_DIR/${dataset}.out $OUTPUT_DIR/${dataset}.id $OUTPUT_DIR/${dataset}.out.post_processed
