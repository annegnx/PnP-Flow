model=ot
eval_split=val
max_batch=5
batch_size_ip=20
problem=denoising

for dataset in celeba afhq_cat
do
method=pnp_flow_grad
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

method=pnp_flow
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.8 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

method=mmse_average
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} max_iter 25 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
done