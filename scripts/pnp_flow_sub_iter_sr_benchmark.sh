problem=superresolution
python main.py --opts method pnp_flow problem $problem steps_pnp 500 sub_iter 1 alpha 0.01 stoppage_iter -1
python main.py --opts method pnp_flow problem $problem steps_pnp 500 sub_iter 10 alpha 0.01 stoppage_iter -1
python main.py --opts method pnp_flow problem $problem steps_pnp 500 sub_iter 50 alpha 0.01 stoppage_iter -1
python main.py --opts method pnp_flow problem $problem steps_pnp 500 sub_iter 100 alpha 0.01 stoppage_iter -1