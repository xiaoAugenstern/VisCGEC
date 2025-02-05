SRC_PATH=./samples/src.txt
HYP_PATH=./samples/hyp.txt
REF_PATH=./samples/ref.txt
OUTPUT_PATH=./samples/output.txt

python pair2edits_char.py $SRC_PATH $HYP_PATH > $OUTPUT_PATH

perl evaluation.pl $OUTPUT_PATH ./demo/report.txt $REF_PATH
