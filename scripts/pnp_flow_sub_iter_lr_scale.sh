problem=inpainting
sub_iter=100
alpha=0.5
stoppage_iter=50
for i in 0.25 0.5 1.0 5.0 10.0 20.0 100.0 1000.0
do
python main.py --opts method pnp_flow problem $problem steps_pnp 100 max_batch 1 batch_size_ip 4 sub_iter $sub_iter stoppage_iter $stoppage_iter lr_scaler $i alpha $alpha
done