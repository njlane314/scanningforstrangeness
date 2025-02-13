from config import Config
from dataset import ContrastiveDataset
from trainers import ContrastiveTrainer

def main():
    config = Config("config/contrastive_config.yaml")
    dataset = ContrastiveDataset(config, background_only=True)
    trainer = ContrastiveTrainer(config, dataset)
    trainer.train()

if __name__ == "__main__":
    main()