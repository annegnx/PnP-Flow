dataset=afhq_cat
model=ot
eval_split=val
max_batch=1
batch_size_ip=4
problem=inpainting

method=pnp_flow_grad
for steps_pnp in 100 200 500 1000
do
for lmbda_0 in 1.0 0.1 0.01
do
for kappa in 10.0 #0.1 0.5 1.0 2.0 5.0
do
for p in 1.1 1.5 1.9
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero lmbda_0 ${lmbda_0} kappa ${kappa} pexp ${p} max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp ${steps_pnp}
done
done
done
done