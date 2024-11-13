dataset=celeba ## or celebahq or afhq_cat
model=ot  ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=test
max_batch=25
batch_size_ip=4



## D FLow
method=d_flow
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 3
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 7
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 10
problem=inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001  alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 9
# problem=paintbrush_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 9
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 20
