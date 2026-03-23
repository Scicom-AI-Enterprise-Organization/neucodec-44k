import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from data_module import DataModule
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import os
import time
import typing
import collections
import omegaconf

from neucodec import NeuCodec
from module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
from criterions import GANLoss, MultiResolutionMelSpectrogramLoss, MultiResolutionSTFTLoss
from common.schedulers import WarmupLR

torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
    collections.defaultdict,
    list,
    dict,
    int,
    omegaconf.dictconfig.DictConfig,
    omegaconf.nodes.AnyNode,
    omegaconf.base.Metadata,
])


seed = 1024
seed_everything(seed)

class CodecLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ocwd = hydra.utils.get_original_cwd()
        self.construct_model()
        self.construct_criteria()
        self.save_hyperparameters()
        self.automatic_optimization = False

    def construct_model(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        decoder_depth = self.cfg.model.codec_decoder.depth
        if self.cfg.ckpt is not None:
            self.model = NeuCodec._from_pretrained(
                model_id=None, local_ckpt_path=self.cfg.ckpt,
                decoder_depth=decoder_depth,
            ).train()
            print(f"Model loaded from local checkpoint: {self.cfg.ckpt}")
        else:
            self.model = NeuCodec._from_pretrained(
                model_id="neuphonic/neucodec",
                decoder_depth=decoder_depth,
            ).train()
            print("Model loaded from HuggingFace")
        for name, p in self.model.named_parameters():
            if 'generator' not in name:
                p.requires_grad = False

        print("Model loaded for finetuning")
        time.sleep(5.0)

        # Initialize MultiPeriod Discriminator
        mpdcfg = self.cfg.model.mpd
        self.discriminator = HiFiGANMultiPeriodDiscriminator(
            periods=mpdcfg.periods,
            max_downsample_channels=mpdcfg.max_downsample_channels,
            channels=mpdcfg.channels,
            channel_increasing_factor=mpdcfg.channel_increasing_factor,
        )

        # Initialize Spectral Discriminator
        mstftcfg = self.cfg.model.mstft
        self.spec_discriminator = SpecDiscriminator(
            stft_params=mstftcfg.stft_params,
            in_channels=mstftcfg.in_channels,
            out_channels=mstftcfg.out_channels,
            kernel_sizes=mstftcfg.kernel_sizes,
            channels=mstftcfg.channels,
            max_downsample_channels=mstftcfg.max_downsample_channels,
            downsample_scales=mstftcfg.downsample_scales,
            use_weight_norm=mstftcfg.use_weight_norm,
        )

        time.sleep(5.0)

    def construct_criteria(self):
        cfg = self.cfg.train
        self.criteria = nn.ModuleDict()
        if cfg.use_mel_loss:
            self.criteria['mel_loss'] = MultiResolutionMelSpectrogramLoss(
                sample_rate=48000,
                window_lengths=list(cfg.mel_window_lengths),
            )
        if cfg.use_stft_loss:
            self.criteria['stft_loss'] = MultiResolutionSTFTLoss(
                fft_sizes=cfg.stft_loss_params.fft_sizes,
                hop_sizes=cfg.stft_loss_params.hop_sizes,
                win_sizes=cfg.stft_loss_params.win_lengths
            )
        if cfg.use_feat_match_loss:
            self.criteria['fm_loss'] = nn.L1Loss()
        self.criteria['gan_loss'] = GANLoss()
        self.criteria['l1_loss'] = nn.L1Loss()
        self.criteria['l2_loss'] = nn.MSELoss()
        print(self.criteria)

    def forward(self, batch):
        wav = batch['wav']  # 16kHz input
        wav_24k = batch['wav_24k']  # 48kHz ground truth
        feats = batch['feats'].squeeze(1)  # pre-computed semantic features from dataloader

        # Encode to FSQ codes (frozen encoder, no gradients needed)
        with torch.no_grad():
            fsq_codes = self.model.encode_code_from_features(wav.unsqueeze(1), feats)

        # Decode back to audio (generator requires gradients for finetuning)
        y_ = self.model.decode_code(fsq_codes)  # (B, 1, T_out)
        y = wav_24k.unsqueeze(1)  # Ground truth with channel dimension

        # Match dimensions if needed
        min_length = min(y_.shape[2], y.shape[2])
        y_ = y_[:, :, :min_length]
        y = y[:, :, :min_length]

        output = {
            'gt_wav': y,
            'gen_wav': y_,
        }
        return output

    def compute_disc_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        y_ = y_.detach()
        p = self.discriminator(y)
        p_ = self.discriminator(y_)

        real_loss_list, fake_loss_list = [], []
        for i in range(len(p)):
            real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(p[i][-1], p_[i][-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        if hasattr(self, 'spec_discriminator'):
            sd_p = self.spec_discriminator(y)
            sd_p_ = self.spec_discriminator(y_)

            for i in range(len(sd_p)):
                real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(sd_p[i][-1], sd_p_[i][-1])
                real_loss_list.append(real_loss)
                fake_loss_list.append(fake_loss)

        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.cfg.train.lambdas.lambda_disc * disc_loss

        output = {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'disc_loss': disc_loss,
        }
        return output

    def compute_gen_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        gen_loss = 0.0
        output_dict = {}
        cfg = self.cfg.train

        # Mel spectrogram loss
        if cfg.use_mel_loss:
            mel_loss = self.criteria['mel_loss'](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * cfg.lambdas.lambda_mel_loss
            output_dict['mel_loss'] = mel_loss

        # GAN loss
        p_ = self.discriminator(y_)
        adv_loss_list = []
        for i in range(len(p_)):
            adv_loss_list.append(self.criteria['gan_loss'].gen_loss(p_[i][-1]))
        if hasattr(self, 'spec_discriminator'):
            sd_p_ = self.spec_discriminator(y_)
            for i in range(len(sd_p_)):
                adv_loss_list.append(self.criteria['gan_loss'].gen_loss(sd_p_[i][-1]))
        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * cfg.lambdas.lambda_adv
        output_dict['adv_loss'] = adv_loss

        # Feature Matching loss
        if cfg.use_feat_match_loss:
            fm_loss = 0.0
            with torch.no_grad():
                p = self.discriminator(y)
            for i in range(len(p_)):
                for j in range(len(p_[i]) - 1):
                    fm_loss += self.criteria['fm_loss'](p_[i][j], p[i][j].detach())
            gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
            output_dict['fm_loss'] = fm_loss
            if hasattr(self, 'spec_discriminator'):
                spec_fm_loss = 0.0
                with torch.no_grad():
                    sd_p = self.spec_discriminator(y)
                for i in range(len(sd_p_)):
                    for j in range(len(sd_p_[i]) - 1):
                        spec_fm_loss += self.criteria['fm_loss'](sd_p_[i][j], sd_p[i][j].detach())
                gen_loss += spec_fm_loss * cfg.lambdas.lambda_feat_match_loss
                output_dict['spec_fm_loss'] = spec_fm_loss

        output_dict['gen_loss'] = gen_loss
        return output_dict

    def training_step(self, batch, batch_idx):
        output = self(batch)

        gen_opt, disc_opt = self.optimizers()
        gen_sche, disc_sche = self.lr_schedulers()

        # Get accumulation steps from config (default to 1)
        accumulate_grad_batches = self.cfg.train.get('accumulate_grad_batches', 1)

        # Determine if this is an accumulation step
        is_accumulating = (batch_idx + 1) % accumulate_grad_batches != 0

        # Zero grads and reset loss accumulators at start of accumulation window
        if batch_idx % accumulate_grad_batches == 0:
            disc_opt.zero_grad()
            gen_opt.zero_grad()
            self._acc_disc_losses = {}
            self._acc_gen_losses = {}

        # Train discriminator
        self.set_discriminator_gradients(True)
        disc_losses = self.compute_disc_loss(batch, output)
        disc_loss = disc_losses['disc_loss'] / accumulate_grad_batches
        self.manual_backward(disc_loss)
        self.set_discriminator_gradients(False)

        # Accumulate disc losses for logging
        for k, v in disc_losses.items():
            if k not in self._acc_disc_losses:
                self._acc_disc_losses[k] = 0.0
            self._acc_disc_losses[k] += v.detach() / accumulate_grad_batches

        # Train generator
        gen_losses = self.compute_gen_loss(batch, output)
        gen_loss = gen_losses['gen_loss'] / accumulate_grad_batches
        self.manual_backward(gen_loss)

        # Accumulate gen losses for logging
        for k, v in gen_losses.items():
            if k not in self._acc_gen_losses:
                self._acc_gen_losses[k] = 0.0
            self._acc_gen_losses[k] += v.detach() / accumulate_grad_batches

        if not is_accumulating:
            self.set_discriminator_gradients(True)
            self.clip_gradients(
                disc_opt,
                gradient_clip_val=self.cfg.train.disc_grad_clip,
                gradient_clip_algorithm='norm'
            )
            disc_opt.step()
            disc_sche.step()

            self.clip_gradients(
                gen_opt,
                gradient_clip_val=self.cfg.train.gen_grad_clip,
                gradient_clip_algorithm='norm'
            )
            gen_opt.step()
            gen_sche.step()

            # Log accumulated losses
            self.log_dict(
                self._acc_disc_losses,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=self.cfg.dataset.train.batch_size,
                sync_dist=True
            )
            self.log_dict(
                self._acc_gen_losses,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=self.cfg.dataset.train.batch_size,
                sync_dist=True
            )

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        from itertools import chain

        # Discriminator parameters
        disc_params = self.discriminator.parameters()
        disc_params = chain(disc_params, self.spec_discriminator.parameters())

        gen_params = (p for p in self.model.parameters() if p.requires_grad)

        # Optimizers
        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)

        # Learning rate schedulers
        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)

        print(f'Generator optim: {gen_opt}')
        print(f'Discriminator optim: {disc_opt}')

        return [gen_opt, disc_opt], [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for p in self.discriminator.parameters():
            p.requires_grad = flag

        if hasattr(self, 'spec_discriminator'):
            for p in self.spec_discriminator.parameters():
                p.requires_grad = flag


@hydra.main(config_path='config', config_name='default', version_base=None)
def train(cfg):
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log_dir, 
        save_top_k=cfg.save_top_k, 
        save_last=True,
        every_n_train_steps=cfg.every_n_train_steps, 
        monitor='step', 
        mode='max',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    datamodule = DataModule(cfg)
    lightning_module = CodecLightningModule(cfg)
    log_dir_name = os.path.basename(os.path.normpath(cfg.log_dir))
    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    ckpt_path = None
    last_ckpt = os.path.join(cfg.log_dir, 'last.ckpt')
    print(last_ckpt)
    if os.path.exists(last_ckpt):
        ckpt_path = last_ckpt
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found, starting training from scratch.")
    
    trainer = pl.Trainer(
        **cfg.train.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=wandb_logger,
        profiler="simple",
        limit_train_batches=1.0 if not cfg.debug else 0.001
    )

    lightning_module = CodecLightningModule(cfg)
    trainer.fit(lightning_module, datamodule=datamodule,ckpt_path=ckpt_path)
    print(f'Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}')


if __name__ == "__main__":
    train()
