problem=superresolution
interpolation_mode='random'
for i in 20 50 80
do
python main.py --opts method pnp_flow problem $problem steps_pnp 100 max_batch 1 batch_size_ip 4 alpha 0.01 sub_iter 100 stoppage_iter $i interpolation_mode $interpolation_mode
done