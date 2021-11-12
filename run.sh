# generate PaT, PaC, TaP triplets
python generate_item_file.py --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode --out-dir=exp

# calculate ABX score using the triplets and codes
python calc_abx.py

# count the mapping between each code and phone
python count_phone_code_map.py --encode-dir=/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode_i --out-dir=exp

# visualize the phone-code mapping as parallel sets diagram
# python visualize_code2counts.py
