python generate_item_file.py \
  --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode_zerospeech2020_LS-test-clean \
  --phone-alignment=phone_alignment.txt \
  --out-dir=exp_zerospeech2020_LS-test-clean

python calc_abx.py --data-dir=exp_zerospeech2020_LS-test-clean

