import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
import pickle as pkl
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import faiss
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Dataset for (img1, img2, label) pairs
class ContrastivePairDataset(Dataset):
    def __init__(self, training_pairs, transform):
        self.pairs = training_pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cand_path, index_path = self.pairs[idx]
        cand_id = os.path.splitext(os.path.basename(cand_path))[0]
        index_id = os.path.splitext(os.path.basename(index_path))[0]
        label = 1 if cand_id == index_id else -1  # CosineEmbeddingLoss expects ±1
        cand_img = self.transform(Image.open(cand_path).convert("RGB"))
        index_img = self.transform(Image.open(index_path).convert("RGB"))
        return cand_img, index_img, torch.tensor(label, dtype=torch.float32)


def get_training_pairs_paths(training_pairs):
    training_pair_paths = []
    for pair in training_pairs:
        training_pair_paths.append((''.join((figs_source_a, f"/{pair[0]}.png")),
                                    ''.join((figs_source_b, f"/{pair[1]}.png"))))
    return training_pair_paths


def get_clip_embedding(model, image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).float()
    return embedding.cpu().numpy()

def get_all_source_embds(figs_dir, rel_ids, model):
    embds = []
    fig_names = os.listdir(figs_dir)
    figs_to_embed = [fig_name for fig_name in fig_names if fig_name.split('.')[0] in rel_ids]
    mapping_dict = {}
    for ind, fig_name in enumerate(tqdm.tqdm(figs_to_embed)):
        embds.append(get_clip_embedding(model, ''.join([figs_dir, '/', fig_name])))
        mapping_dict[ind] = fig_name.split('.')[0]
    embds = np.array(embds, dtype=np.float32)
    return np.squeeze(embds, axis=1), mapping_dict




class ContrastiveModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, img1, img2):
        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)
        return emb1, emb2

# call the main with __name__ == __main__
if __name__ == "__main__":
    seeds_num = 3
    dataset_name = "Hague"
    dataset_partition_path = config.FilePaths.dataset_partition_path
    figs_source_a = 'data/RawCitiesData/The Hague/Source A/png_figs'
    figs_source_b = 'data/RawCitiesData/The Hague/Source B/png_figs'
    # models = ['ViT-B/32', 'ViT-L/14']
    models = ['ViT-L/14']
    contrastive_model_path = 'saved_contrastive_models/'
    if not os.path.exists(contrastive_model_path):
        os.makedirs(contrastive_model_path)

    epochs_dict = {'ViT-B/32': 20, 'ViT-L/14': 8}
    for model_name in models:
        for seed in range(1, seeds_num + 1):
            print(f"model: {model_name}, seed: {seed}")
            dataset_partition_dict = pkl.load(open(f"{dataset_partition_path}{dataset_name}_seed{seed}.pkl", 'rb'))
            training_pairs = dataset_partition_dict['train']['blocking-based']['small'][2]
            model, preprocess = clip.load(model_name, device=device)
            vision_encoder = model.visual
            contrastive_model = ContrastiveModel(vision_encoder).to(device).float()
            loss_fn = nn.CosineEmbeddingLoss()
            optimizer = torch.optim.AdamW(contrastive_model.parameters(), lr=1e-5)
            training_pair_paths = get_training_pairs_paths(training_pairs)
            dataset = ContrastivePairDataset(training_pair_paths, transform=preprocess)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

            epochs = epochs_dict[model_name]
            contrastive_model.train()
            for epoch in range(epochs):
                total_loss = 0
                for img1, img2, label in tqdm.tqdm(dataloader, mininterval=10.0):
                    img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                    emb1, emb2 = contrastive_model(img1, img2)
                    loss = loss_fn(emb1, emb2, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

            # 7. Save the encoder
            safe_model_name = model_name.replace("/", "_")
            full_path = f"{contrastive_model_path}contrastive_model_{safe_model_name}_seed{seed}.pth"
            torch.save(contrastive_model.state_dict(), full_path)

