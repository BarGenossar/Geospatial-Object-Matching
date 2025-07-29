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
from sklearn.metrics import precision_recall_fscore_support
import argparse



device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda:1")


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
        label = 1 if cand_id == index_id else 0
        cand_img = self.transform(Image.open(cand_path).convert("RGB"))
        index_img = self.transform(Image.open(index_path).convert("RGB"))
        return cand_img, index_img, torch.tensor(label, dtype=torch.float32)


def get_training_pairs_paths(training_pairs):
    training_pair_paths = []
    for pair in training_pairs:
        training_pair_paths.append((''.join((figs_source_a, f"/{pair[0]}.png")),
                                    ''.join((figs_source_b, f"/{pair[1]}.png"))))
    return training_pair_paths


def evaluate_model_on_loader(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, label in tqdm.tqdm(dataloader, mininterval=10.0):
            img1, img2 = img1.to(device), img2.to(device)
            output = model(img1, img2)
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    return precision, recall, f1


class ContrastiveBinaryClassifier(nn.Module):
    def __init__(self, encoder, emb_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img1, img2):
        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)
        diff = torch.abs(emb1 - emb2)  # or emb1 * emb2, or torch.cat([...])
        return self.classifier(diff).squeeze(1)


def run_pipeline(model_name, dataset_partition_dict, seed, contrastive_model_path,
                 epochs_dict, clip_embedding_dims, dataset_size):
    training_pairs = dataset_partition_dict['train']['blocking-based'][dataset_size][2]
    model, preprocess = clip.load(model_name, device=device)
    vision_encoder = model.visual
    emb_dim = clip_embedding_dims[model_name]
    contrastive_model = ContrastiveBinaryClassifier(vision_encoder, emb_dim).to(device).float()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(contrastive_model.parameters(), lr=1e-5)
    training_pair_paths = get_training_pairs_paths(training_pairs)
    dataset = ContrastivePairDataset(training_pair_paths, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    best_f1 = -1
    best_model_state = None

    epochs = epochs_dict[model_name]
    contrastive_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for img1, img2, label in tqdm.tqdm(dataloader, mininterval=10.0):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = contrastive_model(img1, img2)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

        precision, recall, f1 = evaluate_model_on_loader(contrastive_model, dataloader, device)
        print(f"[Eval] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = contrastive_model.state_dict()

    safe_model_name = model_name.replace("/", "_")
    best_model_path = f"{contrastive_model_path}best_matching_model_{safe_model_name}_{dataset_size}_seed{seed}.pth"
    torch.save(best_model_state, best_model_path)
    print(f"Best model (F1={best_f1:.4f}) saved to {best_model_path}")
    return run_test(model_name, dataset_partition_dict, seed, contrastive_model_path, dataset_size)


def run_test(model_name, dataset_partition_dict, seed, contrastive_model_path, dataset_size):
    model, preprocess = clip.load(model_name, device=device)
    vision_encoder = model.visual
    emb_dim = clip_embedding_dims[model_name]
    contrastive_model = ContrastiveBinaryClassifier(vision_encoder, emb_dim).to(device).float()
    safe_model_name = model_name.replace("/", "_")
    best_model_path = f"{contrastive_model_path}best_matching_model_{safe_model_name}_{dataset_size}_seed{seed}.pth"
    contrastive_model.load_state_dict(torch.load(best_model_path))
    contrastive_model.eval()

    test_pairs = dataset_partition_dict['test']['matching']['blocking-based'][dataset_size][2]
    test_pair_paths = get_training_pairs_paths(test_pairs)

    dataset = ContrastivePairDataset(test_pair_paths, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    precision, recall, f1 = evaluate_model_on_loader(contrastive_model, dataloader, device)
    print(f"[Test] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {'precision': precision, 'recall': recall, 'f1': f1}


def aggregate_and_save_results(res_dict, model_name, dataset_size, dataset_name):
    precision_list = [res_dict[s]['precision'] for s in res_dict.keys()]
    recall_list = [res_dict[s]['recall'] for s in res_dict.keys()]
    f1_list = [res_dict[s]['f1'] for s in res_dict.keys()]

    agg_dict = {
        'precision': round(np.mean(precision_list), 4),
        'recall': round(np.mean(recall_list), 4),
        'f1': round(np.mean(f1_list), 4)
    }

    output_dir = f"results/matching_baselines/{dataset_name}/"
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_")
    output_path = os.path.join(output_dir, f"{safe_model_name}_vit_with_contrastive_results_dict_{dataset_size}.pkl")
    with open(output_path, 'wb') as f:
        pkl.dump(agg_dict, f)

    print(f"Aggregated results saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=config.Constants.dataset_name)
    parser.add_argument('--seeds_num', type=int, default=3)
    parser.add_argument('--dataset_partition_path', type=str, default=config.FilePaths.dataset_partition_path)
    parser.add_argument('--mode', type=str, default="train+test", help="Mode to run the script in: train+test, test")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    seeds_num = args.seeds_num
    dataset_partition_path = args.dataset_partition_path
    mode = args.mode


    fig_sources_dict = {"Hague": ('data/RawCitiesData/The Hague/Source A/png_figs',
                                  'data/RawCitiesData/The Hague/Source B/png_figs'),
                        "bo_em": ('data/240823/em/png_figs', 'data/240823/bo/png_figs')}
    models = ['ViT-B/32', 'ViT-L/14']
    contrastive_model_path = 'saved_contrastive_models/'
    if not os.path.exists(contrastive_model_path):
        os.makedirs(contrastive_model_path)

    figs_source_a = fig_sources_dict[dataset_name][0]
    figs_source_b = fig_sources_dict[dataset_name][1]
    epochs_dict = {'ViT-B/32': 8, 'ViT-L/14': 5}
    clip_embedding_dims = {'ViT-B/32': 512, 'ViT-L/14': 768}

    for model_name in models:

        #for dataset_size in ['small', 'medium', 'large']:
        for dataset_size in ['small', 'large']:
            res_dict = {}
            for seed in range(1, seeds_num + 1):
                print(f"model: {model_name}, size: {dataset_size}, seed: {seed}")
                dataset_partition_dict = pkl.load(open(f"{dataset_partition_path}{dataset_name}_seed{seed}.pkl", 'rb'))
                if mode == 'train+test':
                    res_dict[seed] = run_pipeline(model_name, dataset_partition_dict, seed,
                                                              contrastive_model_path, epochs_dict,
                                                              clip_embedding_dims, dataset_size)
                elif mode == 'test':
                    res_dict[seed] = run_test(model_name, dataset_partition_dict,
                                                          seed, contrastive_model_path, dataset_size)
            aggregate_and_save_results(res_dict, model_name, dataset_size, dataset_name)






