n_batch=1
batch_size=4
python main.py --opts method map_estimation problem denoising batch_size_ip $batch_size max_batch $n_batch
python main.py --opts method map_estimation problem gaussian_deblurring_FFT batch_size_ip $batch_size max_batch $n_batch
python main.py --opts method map_estimation problem superresolution batch_size_ip $batch_size max_batch $n_batch
python main.py --opts method map_estimation problem random_inpainting batch_size_ip $batch_size max_batch $n_batch
python main.py --opts method map_estimation problem inpainting batch_size_ip $batch_size max_batch $n_batch