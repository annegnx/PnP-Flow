## train OT Flow matching
python main.py --opts dataset afhq_cat train True compute_metrics False batch_size_train 128 num_epoch 600 lr 0.0001 model indep

## train Gradient step denoiser
# python main_denoiser.py --opts train True compute_metrics False model gradient_step dataset afhq_cat batch_size_train 16 num_epoch 200 lr 0.0001