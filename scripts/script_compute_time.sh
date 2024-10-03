dataset=celeba
model=ot
eval_split=test
max_batch=25
batch_size_ip=4
num_samples=5
problem=gaussian_deblurring_FFT

## PNP FLOW
method=pnp_flow
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.01 num_samples ${num_samples} max_batch ${max_batch} batch_size_ip ${batch_size_ip}  compute_memory True compute_time True save_results False steps_pnp 100

## OT ODE
method=ot_ode
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.4  gamma gamma_t max_batch ${max_batch} batch_size_ip ${batch_size_ip}  compute_memory True compute_time True save_results False steps_pnp 100

## FLOW PRIORS
method=flow_priors
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 10000  max_batch ${max_batch} batch_size_ip ${batch_size_ip} compute_memory True compute_time True save_results False

## D FLOW
method=d_flow
python main.py --opts solve_inverse_problem True dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch 50 batch_size_ip 2 max_iter 7 compute_memory True compute_time True save_results False