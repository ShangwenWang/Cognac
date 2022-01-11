export PYTHONPATH=`pwd`
MODEL=$1
python training_ptr_gen/decode_type.py $MODEL >& ../log/decode_log &

