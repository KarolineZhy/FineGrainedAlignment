import lightning as L
from model.ultraclip_lighting import UltraCLIPLightningModule
from dataset.ultrasound_dataset import UltrasoundFinegrainedDataset, DataCollatorUltrasoundFinegrainedDataset
from dataset.mri_dataset import MRIFinegrainedDataset, DataCollatorMRIFinegrainedDataset
from argparse import ArgumentParser
from lightning.pytorch.loggers.wandb import WandbLogger
from pyhocon import ConfigFactory
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.utils import parse_config_to_data_args
from lightning.pytorch.strategies import DeepSpeedStrategy
from dataset.transform import image_transform

if __name__ == "__main__":
    
    # load configuration file
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)
    
    # get data_args for dataset module
    data_args = parse_config_to_data_args(conf.dataset)
    
    wandb.init()
    wandb_logger = WandbLogger()
    
    # seed
    L.seed_everything(conf.train.random_seed)
    
    # init tokenizer and preprocess
    tokenizer = AutoTokenizer.from_pretrained(conf.model.model_name_or_path)
    # img_preprocess = AutoImageProcessor.from_pretrained(conf.model.model_name_or_path)
    img_preprocess_vlm = image_transform(
                image_size=data_args.base_image_size,
                is_train=True,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                fill_color=0)
    
    vfm_patch_size = getattr(conf.model, 'vfm_patch_size', 14)
    vfm_image_size = data_args.base_image_size / 16 * vfm_patch_size
    img_preprocess_vfm = image_transform(
                image_size=vfm_image_size,
                is_train=True,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                fill_color=0)
    
    
    # init dataset
    if conf.dataset.modality == "us":
        train_dataset = UltrasoundFinegrainedDataset(
                        data_path=conf.dataset.data_path,
                        data_args=data_args,
                        img_preprocess_vlm=img_preprocess_vlm,
                        img_preprocess_vfm=img_preprocess_vfm,
                        tokenizer=tokenizer)
        # init dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=conf.dataloader.batch_size, num_workers=conf.dataloader.num_workers, collate_fn=DataCollatorUltrasoundFinegrainedDataset())
    elif conf.dataset.modality in ("mri", "ct"):
        train_dataset = MRIFinegrainedDataset(
                        data_path=conf.dataset.data_path,
                        data_args=data_args,
                        img_preprocess_vlm=img_preprocess_vlm,
                        img_preprocess_vfm=img_preprocess_vfm,
                        tokenizer=tokenizer)        
        # init dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=conf.dataloader.batch_size, num_workers=conf.dataloader.num_workers, collate_fn=DataCollatorMRIFinegrainedDataset())
    else:
        raise ValueError(f"Invalid modality: {conf.dataset.modality}")
    print('train dataset length: ', len(train_dataset))
    
    

    # init checkpoint callback and save model every 5 epochs
    checkpoint_callback = ModelCheckpoint(
    dirpath=conf.train.checkpoint_dir,
    # filename="checkpoint-{epoch:02d}",
    every_n_epochs=conf.train.save_epoch_freq,             
    save_top_k=-1,               
    save_weights_only=False,
    verbose=True,
    )

    # init model
    model = UltraCLIPLightningModule(conf)
    
    # init trainer
    # trainer = L.Trainer(
    #     max_epochs=conf.train.max_steps,
    #     accelerator=conf.train.accelerator,
    #     precision=conf.train.precision,
    #     logger=wandb_logger,
    #     callbacks=[checkpoint_callback],
    #     strategy='ddp'
    # )
    
    trainer = L.Trainer(
        accelerator=conf.train.accelerator,
        precision=conf.train.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=conf.train.max_epochs,
        strategy="ddp_find_unused_parameters_true"
    )

    # train
    trainer.fit(model, train_dataloader)


    