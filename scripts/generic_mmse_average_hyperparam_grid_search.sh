dataset=celeba
model=ot
eval_split=val
max_batch=1
batch_size_ip=4
steps_pnp=100
lmbda_0=0.0
problem=inpainting

method=generic_mmse_average
for b_1 in 1.0 #0.2 0.3 0.5 0.6 0.8 1.0 1.2 1.5 2.0
do
for b_2 in 1.3 #0.1 0.2 0.5 0.8 1.0 1.5 2.0
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero lmbda_0 ${lmbda_0} b_1 ${b_1} b_2 ${b_2} max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp ${steps_pnp}
done
done