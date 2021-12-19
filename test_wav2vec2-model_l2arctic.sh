exp_dir=exp_wav2vec2_l2arctic

python generate_item_file.py \
  --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/wav2vec2/encode_wav2vec2_l2arctic \
  --phone-alignment=phone_alignment.txt \
  --freq=49 \
  --out-dir=${exp_dir}

python calc_abx.py --data-dir=${exp_dir}
