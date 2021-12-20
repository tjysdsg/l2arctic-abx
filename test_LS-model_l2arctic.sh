python generate_item_file.py \
  --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode_librispeech960_l2arctic \
  --phone-alignment=phone_alignment.txt \
  --model-config=config/zerospeech2020.json \
  --sample-rate=16000 \
  --hop-length=160 \
  --out-dir=exp_LS_l2arctic

python calc_abx.py --data-dir=exp_LS_l2arctic

python count_phone_code_map.py \
  --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode_index_librispeech960_l2arctic \
  --phone-alignment=phone_alignment.txt \
  --out-dir=exp_LS_l2arctic
