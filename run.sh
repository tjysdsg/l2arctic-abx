# generate PaT, PaC, TaP triplets
python generate_item_file.py \
  --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode_librispeech960_l2arctic \
  --phone-alignment=phone_alignment.txt \
  --out-dir=exp

# calculate ABX score using the triplets and codes
python calc_abx.py --data-dir=exp

# count the mapping between each code and phone
python count_phone_code_map.py \
  --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode_librispeech960_l2arctic_i \
  --phone-alignment=phone_alignment.txt \
  --out-dir=exp

# visualize the phone-code mapping as parallel sets diagram
# python visualize_code2counts.py

