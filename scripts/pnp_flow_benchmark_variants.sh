problem=denoising
python main.py --opts method pnp_flow problem $problem steps_pnp 100 alpha 0.8 interpolation_mode random datafit_mode gd denoise_mode gd
python main.py --opts method pnp_flow problem $problem steps_pnp 100 alpha 0.8 interpolation_mode random datafit_mode prox denoise_mode gd
python main.py --opts method pnp_flow problem $problem steps_pnp 100 alpha 0.8 interpolation_mode random datafit_mode gd denoise_mode prox
python main.py --opts method pnp_flow problem $problem steps_pnp 100 alpha 0.8 interpolation_mode random datafit_mode prox denoise_mode prox

problem=superresolution
python main.py --opts method pnp_flow problem $problem steps_pnp 500 alpha 0.01 interpolation_mode random datafit_mode gd denoise_mode gd
python main.py --opts method pnp_flow problem $problem steps_pnp 500 alpha 0.01 interpolation_mode random datafit_mode prox denoise_mode gd
python main.py --opts method pnp_flow problem $problem steps_pnp 500 alpha 0.01 interpolation_mode random datafit_mode gd denoise_mode prox
python main.py --opts method pnp_flow problem $problem steps_pnp 500 alpha 0.01 interpolation_mode random datafit_mode prox denoise_mode prox

problem=inpainting
python main.py --opts method pnp_flow problem $problem steps_pnp 100 alpha 0.5 interpolation_mode random datafit_mode gd denoise_mode gd
python main.py --opts method pnp_flow problem $problem steps_pnp 100 alpha 0.5 interpolation_mode random datafit_mode prox denoise_mode gd
python main.py --opts method pnp_flow problem $problem steps_pnp 100 alpha 0.5 interpolation_mode random datafit_mode gd denoise_mode prox
python main.py --opts method pnp_flow problem $problem steps_pnp 100 alpha 0.5 interpolation_mode random datafit_mode prox denoise_mode prox
