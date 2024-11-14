dataset=celeba ## or celebahq or afhq_cat
model=ot  ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=val
max_batch=8
batch_size_ip=4

### PNP FLOW
method=pnp_flow
for alpha in 0.01 0.1 0.3 0.5 0.8 1.0  #1.2 1.5 1.7 1.9 2.2 #
do
for problem in superresolution
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha ${alpha} num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100 noise_type laplace
done
done


# ### FLOW PRIORS
# method=flow_priors
# for lmbda in 100 1000 10000 100000
# do
# for eta in 0.001 0.01 0.1
# do
# for problem in superresolution
# do
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta ${eta} lmbda ${lmbda} max_batch ${max_batch} batch_size_ip ${batch_size_ip} noise_type laplace
# done
# done
# done


# ### PNP GRADIENT STEP
# method=pnp_gs
# model=gradient_step
# for lr_pnp in 0.99 2.
# do
# for alpha in 0.3 0.5 0.8 1.0
# do
# for sigma_factor in 1. 1.2 1.5 1.8 2. 3. 4. 5. 6. 8. 10.
# do
# for problem in superresolution
# do
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp ${lr_pnp} alpha ${alpha} max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 100 noise_type laplace sigma_factor ${sigma_factor}
# done
# done
# done
# done


# ### PNP DIFFUSION
# method=pnp_diff
# model=diffusion
# for lmbda in 1.0 5.0 10.0 100.0 1000.0
# do
# for zeta in 0.1 0.3 0.5 1.0
# do
# for problem in superresolution
# do
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda ${lmbda} zeta ${zeta} max_batch ${max_batch} batch_size_ip ${batch_size_ip} noise_type laplace
# done
# done
# done