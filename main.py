from model import StandardAutoregressiveModel, generate_square_subsequent_mask
from trainer import train_model
from data_preparation import get_dataloader

if __name__ == '__main__':
  train_model()
