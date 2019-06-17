import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

# celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
#                            config.celeba_crop_size, config.image_size, config.batch_size,
#                            'CelebA', config.mode, config.num_workers)

class Alt_config():
    def __init__(self):
        # self.dataset = 'CelebA'
        # self.celeba_image_dir
        # self.attr_path
        # self.selected_attrs
        # self.celeba_crop_size
        # self.image_size
        # self.batch_size
        # self.mode
        # self.num_workers

        # Model configuration.
        self.c_dim=5
        self.c2_dim=8
        self.celeba_crop_size=178
        self.rafd_crop_size=256
        self.image_size=128
        self.g_conv_dim=64
        self.d_conv_dim=64
        self.g_repeat_num=6
        self.d_repeat_num=6
        self.lambda_cls=1
        self.lambda_rec=10
        self.lambda_gp=10

        # Training configuration.
        self.dataset='CelebA'
        self.batch_size=16
        self.num_iters=200000
        self.num_iters_decay=100000
        self.g_lr=0.0001
        self.d_lr=0.0001
        self.n_critic=5
        self.beta1=0.5
        self.beta2=0.999
        self.resume_iters=None
        self.selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

        # Test configuration.
        self.test_iters=200000

        # Miscellaneous.
        self.num_workers=1
        self.mode='test'
        self.use_tensorboard=True

        # Directories.
        self.celeba_image_dir='data/celeb_test_custom/images'
        self.attr_path='data/celeb_test_custom/list_attr_celeba.txt'
        self.rafd_image_dir='data/RaFD/train'
        self.log_dir='stargan/logs'
        self.model_save_dir='stargan_celeba_256/models'
        self.sample_dir='stargan/samples'
        self.result_dir='stargan/test_results'

        # Step size.
        self.log_step=10
        self.sample_step=1000
        self.model_save_step=10000
        self.lr_update_step=1000





def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None


    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)


    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    config = Alt_config()
    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                               config.celeba_crop_size, config.image_size, config.batch_size,
                               'CelebA', config.mode, config.num_workers)
    solver = Solver(celeba_loader, None, config)
    solver.test_single()



    main(config)
