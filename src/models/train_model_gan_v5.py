import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import click
import torch.nn as nn
import torch.optim as optim
import logging
from tempfile import NamedTemporaryFile, TemporaryDirectory
from pathlib import Path
import sys
from argparse import Namespace
from src.data.dataset import ToTensor, Normalize, TEPDatasetV4, InverseNormalize
from src.models.utils import get_latest_model_id
from src.models.recurrent_models import TEPRNN, LSTMGenerator
from src.models.convolutional_models import (
    # CausalConvDiscriminator,
    # CausalConvGenerator,
    CausalConvDiscriminatorMultitask
)
import torch.backends.cudnn as cudnn
from src.data.dataset import TEPDataset
from src.models.utils import time_series_to_plot
from tensorboardX import SummaryWriter
import random
from PIL import Image

"""This is training of Conditioned GAN model script."""

REAL_LABEL = 1
FAKE_LABEL = 0


@click.command()
@click.option('--cuda', required=True, type=int, default=7)
@click.option('-d', '--debug', 'debug', is_flag=True)
@click.option('--run_tag', required=True, type=str, default="unknown")
@click.option('--random_seed', required=False, type=int, default=None)
def main(cuda, debug, run_tag, random_seed):
    """
    todo: write something
    """
    # for tensorboard logs
    try:
        os.makedirs("logs")
    except OSError:
        pass

    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.INFO)
    temp_model_dir = TemporaryDirectory(dir="models")
    temp_model_dir.cleanup()
    Path(temp_model_dir.name).mkdir(parents=True, exist_ok=False)
    Path(os.path.join(temp_model_dir.name), "images").mkdir(parents=True, exist_ok=False)
    Path(os.path.join(temp_model_dir.name), "weights").mkdir(parents=True, exist_ok=False)
    temp_model_dir_tensorboard = TemporaryDirectory(dir="logs")
    temp_model_dir_tensorboard.cleanup()
    Path(temp_model_dir_tensorboard.name).mkdir(parents=True, exist_ok=False)
    temp_log_file = os.path.join(temp_model_dir.name, 'log.txt')
    file_handler = logging.FileHandler(temp_log_file)
    file_handler.setFormatter(log_formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f'Training begin on {device}')
    logger.info(f'Tmp TB dir {temp_model_dir_tensorboard.name}, tmp model dir {temp_model_dir.name}')

    if random_seed is None:
        random_seed = random.randint(1, 10000)
    logger.info(f"Random Seed: {random_seed}")
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    cudnn.benchmark = True

    # todo: add this nice syntax
    flags = Namespace(
        train_file='oliver.txt',
        seq_size=32,
        batch_size=16,
        embedding_size=64,
        lstm_size=64,
        gradients_norm=5,
        initial_words=['I', 'am'],
        predict_top_k=5,
        checkpoint_path='checkpoint',
    )

    lstm_size = 64
    loader_jobs = 1
    epochs = 50
    window_size = 30
    bs = 128
    tep_file_fault_free_train = "data/raw/TEP_FaultFree_Training.RData"
    tep_file_faulty_train = "data/raw/TEP_Faulty_Training.RData"
    tep_file_fault_free_test = "data/raw/TEP_FaultFree_Testing.RData"
    tep_file_faulty_test = "data/raw/TEP_Faulty_Testing.RData"
    noise_size = 100
    conditioning_size = 1
    in_dim = noise_size + conditioning_size
    fault_type_classifier_weights = "models/2/latest.pth"
    checkpoint_every = 10
    real_fake_w_d = 1.0  # weight for real fake in loss
    fault_type_w_d = 0.01  # weight for fault type term in loss
    real_fake_w_g = 1.0  # weight for real fake in loss
    similarity_w_g = 0.2  # weight for fault type term in loss

    if debug:
        loader_jobs = 1
        bs = 2
        epochs = 4
        tep_file_fault_free_train = "data/raw/sampled_TEP/sampled_train.pkl"
        tep_file_faulty_train = "data/raw/sampled_TEP/sampled_train.pkl"
        tep_file_fault_free_test = "data/raw/sampled_TEP/sampled_test.pkl"
        tep_file_faulty_test = "data/raw/sampled_TEP/sampled_test.pkl"
        checkpoint_every = 1

    # Create writer for tensorboard
    writer = SummaryWriter(temp_model_dir_tensorboard.name)
    # todo: add all running options to some structure and print them.
    writer.add_text('Options', str("Here yo can write running options or put your ads."), 0)

    transform = transforms.Compose([
        ToTensor(),
        Normalize()
    ])

    inverse_transform = InverseNormalize()

    trainset = TEPDatasetV4(
        tep_file_fault_free=tep_file_fault_free_train,
        tep_file_faulty=tep_file_faulty_train,
        # window_size=window_size,
        is_test=False,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=loader_jobs,
                                              drop_last=False)

    printset = TEPDatasetV4(
        tep_file_fault_free=tep_file_fault_free_train,
        tep_file_faulty=tep_file_faulty_train,
        # window_size=window_size,
        is_test=False,
        transform=None,
        for_print=True
    )

    printloader = torch.utils.data.DataLoader(printset, batch_size=trainset.class_count, shuffle=False, num_workers=1,
                                              drop_last=False)

    # netD = CausalConvDiscriminator(input_size=trainset.features_count,
    #                                n_layers=8, n_channel=10, kernel_size=8,
    #                                dropout=0).to(device)
    netG = LSTMGenerator(in_dim=in_dim, out_dim=52, hidden_dim=256).to(device)
    # netG = CausalConvGenerator(noise_size=in_dim, output_size=52, n_layers=8, n_channel=10, kernel_size=8,
    #                            dropout=0.2).to(device)

    netD = CausalConvDiscriminatorMultitask(input_size=trainset.features_count,
                                            n_layers=8, n_channel=10, class_count=trainset.class_count,
                                            kernel_size=8, dropout=0).to(device)

    logger.info("Generator:\n" + str(netG))
    logger.info("Discriminator:\n" + str(netD))

    binary_criterion = nn.BCEWithLogitsLoss()
    cross_entropy_criterion = nn.CrossEntropyLoss()
    similarity = nn.MSELoss(reduction='sum')
    # similarity = nn.L1Loss(reduction='sum')


    optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002)

    for epoch in range(epochs):

        logger.info('Epoch %d training...' % epoch)
        netD.train()
        netG.train()
        # can do this cause we dont use optimizer for this net now

        for i, data in enumerate(trainloader, 0):
            n_iter = epoch * len(trainloader) + i

            real_inputs, fault_labels = data["shot"], data["label"]
            real_inputs, fault_labels = real_inputs.to(device), fault_labels.to(device)
            real_inputs = real_inputs.squeeze(dim=1)
            fault_labels = fault_labels.squeeze()

            netD.zero_grad()

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # Real data training
            batch_size, seq_len = real_inputs.size(0), real_inputs.size(1)
            real_target = torch.full((batch_size, seq_len, 1), REAL_LABEL, device=device)

            type_logits, fake_logits = netD(real_inputs, None)
            errD_real = binary_criterion(fake_logits, real_target)
            errD_type_real = cross_entropy_criterion(type_logits.transpose(1, 2), fault_labels)
            # todo: print at tensorboard errD_real + errD_type

            errD_complex_real = real_fake_w_d * errD_real + fault_type_w_d * errD_type_real
            errD_complex_real.backward()
            # here is missed sigmoid function
            D_x = type_logits.mean().item()

            # Fake data training
            noise = torch.randn(batch_size, seq_len, noise_size, device=device)
            random_labels = torch.randint(high=trainset.class_count, size=(batch_size, 1, 1),
                                          dtype=torch.float32, device=device)
            random_labels = random_labels.repeat(1, seq_len, 1)
            random_labels[:, :20, :] = 0
            noise = torch.cat((noise, random_labels), dim=2)

            state_h, state_c = netG.zero_state(batch_size)
            state_h, state_c = state_h.to(device), state_c.to(device)

            fake_inputs = netG(noise, (state_h, state_c))
            fake_target = torch.full((batch_size, seq_len, 1), FAKE_LABEL, device=device)
            # WARNING: do not forget about detach!
            type_logits, fake_logits = netD(fake_inputs.detach(), None)
            errD_fake = binary_criterion(fake_logits, fake_target)
            errD_type_fake = cross_entropy_criterion(type_logits.transpose(1, 2), random_labels.long().squeeze())

            errD_complex_fake = real_fake_w_d * errD_fake + fault_type_w_d * errD_type_fake
            errD_complex_fake.backward()
            # here is missed sigmoid function
            D_G_z1 = type_logits.mean().item()
            # all good here (2 lines)
            errD = real_fake_w_d * errD_real + real_fake_w_d * errD_fake
            errD_fault_type = fault_type_w_d * errD_type_real + fault_type_w_d * errD_type_fake

            optimizerD.step()


            # Visualize discriminator gradients
            for name, param in netD.named_parameters():
                writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, n_iter)

            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            type_logits, fake_logits = netD(fake_inputs, None)
            errG = real_fake_w_g * binary_criterion(fake_logits, real_target)
            errG.backward()
            D_G_z2 = fake_logits.mean().item()

            # Visual similarity correction
            # todo: add KL-divergence term
            noise = torch.randn(batch_size, seq_len, noise_size, device=device)
            noise = torch.cat((noise, fault_labels.float().unsqueeze(2)), dim=2)

            state_h, state_c = netG.zero_state(batch_size)
            state_h, state_c = state_h.to(device), state_c.to(device)
            out_seqs = netG(noise, (state_h, state_c))

            errG_similarity = similarity_w_g * similarity(out_seqs, real_inputs)
            errG_similarity.backward()

            optimizerG.step()

            # Visualize generator gradients
            for name, param in netG.named_parameters():
                writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, n_iter)

            log_flag = True if debug else (i + 1) % 20 == 0
            if log_flag:
                logger.info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' %
                            (epoch, epochs, i, len(trainloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            writer.add_scalar('DiscriminatorLoss', errD.item(), n_iter)
            writer.add_scalar('DiscriminatorLossFaultType', errD_fault_type.item(), n_iter)
            writer.add_scalar('GeneratorLoss', errG.item(), n_iter)
            writer.add_scalar('GeneratorLossSimilarity', errG_similarity.item(), n_iter)
            writer.add_scalar('DofX_noSigmoid', D_x, n_iter)
            writer.add_scalar('DofGofz_noSigmoid', D_G_z1, n_iter)

            if debug and i > 1:
                break

        logger.info('Epoch %d passed' % epoch)

        # Saving epoch results.
        # The following images savings cost a lot of memory, reduce the frequency
        if epoch in [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, epochs - 1] and not debug:
            netD.eval()
            netG.eval()

            real_display = next(iter(printloader))
            real_inputs, true_labels = real_display["shot"], real_display["label"]
            real_inputs, true_labels = real_inputs.to(device), true_labels.to(device)

            real_display = inverse_transform(real_display)

            real_plots = time_series_to_plot(real_display["shot"].cpu())
            for idx, rp in enumerate(real_plots):
                fp_real = os.path.join(temp_model_dir.name, "images", f"{epoch}_epoch_real_{idx}.jpg")
                ndarr = rp.to('cpu', torch.uint8).permute(1, 2, 0).numpy()
                im = Image.fromarray(ndarr, mode="RGB")
                im.save(fp_real, format=None)
                writer.add_image(f"RealTEP_{idx}", rp, epoch)

            batch_size, seq_len = real_inputs.size(0), real_inputs.size(1)
            noise = torch.randn(batch_size, seq_len, noise_size, device=device)
            noise = torch.cat((noise, true_labels.float()), dim=2)

            state_h, state_c = netG.zero_state(batch_size)
            state_h, state_c = state_h.to(device), state_c.to(device)
            fake_display = netG(noise, (state_h, state_c))
            fake_display = {"shot": fake_display.cpu(), "label": true_labels}
            fake_display = inverse_transform(fake_display)

            fake_plots = time_series_to_plot(fake_display["shot"])
            for idx, fp in enumerate(fake_plots):
                fp_fake = os.path.join(temp_model_dir.name, "images", f"{epoch}_epoch_fake_{idx}.jpg")
                ndarr = fp.to('cpu', torch.uint8).permute(1, 2, 0).numpy()
                im = Image.fromarray(ndarr, mode="RGB")
                im.save(fp_fake, format=None)
                writer.add_image(f"FakeTEP_{idx}", fp, epoch)

        if (epoch % checkpoint_every == 0) or (epoch == (epochs - 1)):
            torch.save(netG, os.path.join(temp_model_dir.name, "weights", f"{epoch}_epoch_generator.pth"))
            torch.save(netD, os.path.join(temp_model_dir.name, "weights", f"{epoch}_epoch_discriminator.pth"))

        printset.change_print_sim_run()

    logger.info(f'Finished training for {epochs} epochs.')

    file_handler.close()
    writer.close()

    with open(__file__, 'r') as f:
        with open(os.path.join(temp_model_dir.name, "script.py"), 'w') as out:
            print("# This file was saved automatically during the experiment run.\n", end='', file=out)
            for line in f.readlines():
                print(line, end='', file=out)

    latest_model_id = get_latest_model_id(dir_name="models") + 1
    os.rename(
        temp_model_dir.name,
        os.path.join("models", f'{latest_model_id}_{run_tag}')
    )

    os.rename(
        temp_model_dir_tensorboard.name,
        os.path.join("logs", f'{latest_model_id}_{run_tag}')
    )


if __name__ == '__main__':
    main()
