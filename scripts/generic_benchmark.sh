model=ot
eval_split=val
max_batch=5
batch_size_ip=20

#gaussian deblurring
problem=gaussian_deblurring_FFT
lmbda_0=0.01
kappa=2.0
pexp=1.9

dataset=celeba
method=pnp_flow_grad
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp 50 max_batch ${max_batch} batch_size_ip ${batch_size_ip} lmbda_0 ${lmbda_0} kappa ${kappa} pexp ${pexp}

#method=pnp_flow
#python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

dataset=afhq_cat
method=pnp_flow_grad
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp 50 max_batch ${max_batch} batch_size_ip ${batch_size_ip} lmbda_0 ${lmbda_0} kappa ${kappa} pexp ${pexp}

#method=pnp_flow
#python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 500 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


#superresolution
problem=superresolution
lmbda_0=0.01
kappa=5.0
pexp=1.9

dataset=celeba
method=pnp_flow_grad
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp 500 max_batch ${max_batch} batch_size_ip ${batch_size_ip} lmbda_0 ${lmbda_0} kappa ${kappa} pexp ${pexp}

#method=pnp_flow
#python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.3 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

dataset=afhq_cat
method=pnp_flow_grad
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp 500 max_batch ${max_batch} batch_size_ip ${batch_size_ip} lmbda_0 ${lmbda_0} kappa ${kappa} pexp ${pexp}

#method=pnp_flow
#python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 500 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


#random_inpainting
problem=random_inpainting
lmbda_0=0.01
kappa=1.0
pexp=1.9

dataset=celeba
method=pnp_flow_grad
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp 300 max_batch ${max_batch} batch_size_ip ${batch_size_ip} lmbda_0 ${lmbda_0} kappa ${kappa} pexp ${pexp}

#method=pnp_flow
#python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

dataset=afhq_cat
method=pnp_flow_grad
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp 300 max_batch ${max_batch} batch_size_ip ${batch_size_ip} lmbda_0 ${lmbda_0} kappa ${kappa} pexp ${pexp}

method=pnp_flow
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 200 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


#inpainting
problem=inpainting
lmbda_0=0.01
kappa=5.0
pexp=1.9

dataset=celeba
method=pnp_flow_grad
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip} lmbda_0 ${lmbda_0} kappa ${kappa} pexp ${pexp}

method=pnp_flow
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.5 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

dataset=afhq_cat
method=pnp_flow_grad
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip} lmbda_0 ${lmbda_0} kappa ${kappa} pexp ${pexp}

method=pnp_flow
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.5 steps_pnp 500 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
