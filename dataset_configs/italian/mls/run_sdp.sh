SDP_ROOT="$HOME/exp/NeMo-speech-data-processor/"

HYDRA_FULL_ERROR=1 PYTHONPATH="/home/erastorgueva/exp/23_01/spanish_pc/mls" python $SDP_ROOT/main.py \
    --config-path="/home/erastorgueva/exp/23_01/spanish_pc/mls" \
    --config-name="config.yaml"
