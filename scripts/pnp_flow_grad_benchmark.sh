n_batch=1
batch_size=4
interpolation_mode=random
num_samples=5
python main.py --opts method pnp_flow_grad steps_pnp 100 interpolation_mode $interpolation_mode num_samples $num_samples alpha 1.0 batch_size_ip $batch_size max_batch $n_batch problem denoising
python main.py --opts method pnp_flow_grad steps_pnp 500 interpolation_mode $interpolation_mode num_samples $num_samples alpha 0.5 batch_size_ip $batch_size max_batch $n_batch problem gaussian_deblurring_FFT
python main.py --opts method pnp_flow_grad steps_pnp 500 interpolation_mode $interpolation_mode num_samples $num_samples alpha 0.1 batch_size_ip $batch_size max_batch $n_batch problem superresolution
python main.py --opts method pnp_flow_grad steps_pnp 200 interpolation_mode $interpolation_mode num_samples $num_samples alpha 0.01 batch_size_ip $batch_size max_batch $n_batch problem random_inpainting
python main.py --opts method pnp_flow_grad steps_pnp 100 interpolation_mode $interpolation_mode num_samples $num_samples alpha 1.0 batch_size_ip $batch_size max_batch $n_batch problem inpainting