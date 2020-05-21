#!/usr/bin/env bash
/Users/eric/byted/bin/python test.py --dataroot database/horse2zebra/valA \
  --dataset_mode url \
  --results_dir results-pretrained/cycle_gan/horse2zebra/compressed \
  --config_str 16_16_32_16_32_32_16_16 \
  --restore_G_path pretrained/cycle_gan/horse2zebra/compressed/latest_net_G.pth \
  --need_profile \
  --no_fid \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --max_dataset_size -1 \
  --input_urls https://storage.googleapis.com/ylq_server/test.jpg
