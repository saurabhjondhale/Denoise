from torch.utils.data import Dataset
import cv2

class DenoisingDataset(Dataset):
    def __init__(self, clean_images, noise_type="gaussian"):
        self.clean_images = clean_images
        self.noise_type = noise_type

    def __getitem__(self, idx):
        clean = self.clean_images[idx]
        noisy = add_noise(clean, self.noise_type)
        return noisy, clean