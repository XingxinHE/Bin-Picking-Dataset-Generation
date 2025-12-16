```
uv run 1_pybullet_create_n_collect.py \
    --start_cycle 1 \
    --end_cycle 16 \
    --mode direct \
    --renderer tiny \
    --workers 16 \
    --dataset_name S_packing_16cycles_10drops \
    --model_name teris \
    --dropping packing \
    --max_drop 10 \
    --object_types S
```


```
uv run 4_generate_h5.py \
    --dataset_name S_packing_16cycles_10drops \
    --workers 16
```