## train OT Flow matching
python main.py --opts dataset afhq_cat train True compute_metrics False batch_size_train 32 num_epoch 400 lr 0.0001 model ot

## train Gradient step denoiser
# python main_denoiser.py --opts train True compute_metrics False model gradient_step dataset afhq_cat batch_size_train 16 num_epoch 200 lr 0.0001