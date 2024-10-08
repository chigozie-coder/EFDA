# training.py

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from model import StandardAutoregressiveModel, generate_square_subsequent_mask
from data_preparation import get_dataloader

class LightningTrainer(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-4):
        super(LightningTrainer, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, tgt, tgt_mask=None):
        return self.model(tgt, tgt_mask)

    def training_step(self, batch, batch_idx):
        tgt_input = batch['input_ids']
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)
        outputs = self(tgt_input, tgt_mask)
        
        loss = self.model.compute_loss(outputs, tgt_input)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tgt_input = batch['input_ids']
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)
        outputs = self(tgt_input, tgt_mask)
        
        loss = self.model.compute_loss(outputs, tgt_input)
        self.log('val_loss', loss)

    def train_dataloader(self):
      return self.train_data

    def val_dataloader(self):
      return self.val_data

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate, steps_per_epoch=len(self.train_dataloader()), epochs=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def train_model():
    train_data = get_dataloader(batch_size=32, shuffle=True)
    val_data = get_dataloader(batch_size=32, shuffle=False, is_validation=True)  # Add validation loader

    num_tokens = train_data.dataset.vocab_size
    d_model = 256
    nhead = 8
    num_decoder_layers = 6
    dim_feedforward = 1024

    model = StandardAutoregressiveModel(num_tokens, d_model, nhead, num_decoder_layers, dim_feedforward)

    trainer_module = LightningTrainer(model)
    trainer_module.train_data = train_data
    trainer_module.val_data = val_data

    # Use bfloat16 precision, accumulate gradients over 2 batches, and stop early if no improvement
    trainer = pl.Trainer(
        max_epochs=3, 
        precision="bf16",  # Use bfloat16 precision for efficient training
        gradient_clip_val=0.5,
        accumulate_grad_batches=2,  # Accumulate gradients to simulate larger batch sizes
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=3),  # Early stopping
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss', 
                save_top_k=1, 
                mode='min', 
                dirpath='checkpoints/', 
                filename='best_model'
            )
        ]  # Model checkpointing
    )
    trainer.fit(trainer_module)  # Pass both train and val data
    torch.save(model.state_dict(), "model.pt")