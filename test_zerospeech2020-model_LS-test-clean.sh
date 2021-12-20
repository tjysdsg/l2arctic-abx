python generate_item_file.py \
  --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode_zerospeech2020_LS-test-clean \
  --utt2spk=/home/storage15/tangjiyang/librispeech_ali_test_clean/kaldi_format/utt2spk \
  --phone-alignment=/home/storage15/tangjiyang/librispeech_ali_test_clean/align.txt \
  --model-config=config/zerospeech2020.json \
  --sample-rate=16000 \
  --hop-length=160 \
  --out-dir=exp_zerospeech2020_LS-test-clean

python calc_abx.py --data-dir=exp_zerospeech2020_LS-test-clean

python count_phone_code_map.py \
  --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode_index_zerospeech2020_LS-test-clean \
  --phone-alignment=/home/storage15/tangjiyang/librispeech_ali_test_clean/align.txt \
  --out-dir=exp_zerospeech2020_LS-test-clean
