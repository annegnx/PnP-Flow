dataset=celeba ## or celebahq or afhq_cat
model=ot  ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=test
max_batch=25
batch_size_ip=4

### PNP FLOW
# method=pnp_flow
# problem=denoising
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 1.7 num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100 noise_type laplace
# problem=gaussian_deblurring_FFT
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 1.7 num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100 noise_type laplace
# problem=superresolution
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.3 num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100
# problem=inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.5 num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100
# problem=paintbrush_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.5 num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100
# problem=random_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.01 num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100



# ### OT ODE
# method=ot_ode
# problem=denoising
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.3 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma gamma_t
# problem=gaussian_deblurring_FFT
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.4 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma gamma_t
# problem=superresolution
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma constant
# problem=inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.1  max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma gamma_t
# problem=paintbrush_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma gamma_t
# problem=random_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma constant


### FLOW PRIORS
method=flow_priors
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip} noise_type laplace
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 1000 max_batch ${max_batch} batch_size_ip ${batch_size_ip} noise_type laplace
# problem=superresolution
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.1 lmbda 10000 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
# problem=inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 10000 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
# # problem=paintbrush_inpainting
# # python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 10000 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
# problem=random_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 10000 max_batch ${max_batch} batch_size_ip ${batch_size_ip}



# ## D FLow
# method=d_flow
# problem=denoising
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 3
# problem=gaussian_deblurring_FFT
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 7
# problem=superresolution
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 10
# problem=inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001  alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 9
# problem=paintbrush_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 9
# problem=random_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 20



# ### PNP GRADIENT STEP
# method=pnp_gs
# model=gradient_step
# problem=denoising
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 1 sigma_factor 1.0
# problem=gaussian_deblurring_FFT
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 2.0 alpha 0.5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 35 sigma_factor 1.8
# problem=superresolution
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 2.0 alpha 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 20 sigma_factor 1.8
# problem=random_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo hqs max_iter 20


# ### PNP DIFFUSION
# method=pnp_diff
# model=diffusion
# problem=denoising
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 1.0 zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
# problem=gaussian_deblurring_FFT
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 1000.0 zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
# problem=superresolution
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 100.0 zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
# problem=random_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 1.0 zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

