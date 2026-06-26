problem=random_inpainting
python main.py --opts method pnp_flow problem $problem steps_pnp 200 alpha 0.01 interpolation_mode random
python main.py --opts method pnp_flow problem $problem steps_pnp 200 alpha 0.01 interpolation_mode zero
python main.py --opts method pnp_flow problem $problem steps_pnp 200 alpha 0.01 interpolation_mode fixed
python main.py --opts method pnp_flow problem $problem steps_pnp 200 alpha 0.01 interpolation_mode id