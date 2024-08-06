```python
import random
from logging import getLogger
from PIL import ImageFilter
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision

logger = getLogger()

# PIL Random Gaussian Blur Class
class PILRandomGaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

# Get color distortion transform
def get_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

# MultiCropDataset class
class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops

# Example usage
if __name__ == "__main__":
    data_path = "/content/train"  # Modify to your dataset path
    size_crops = [224, 96]
    nmb_crops = [2, 3]
    min_scale_crops = [0.6, 0.2]
    max_scale_crops = [1.0, 0.8]

    dataset = MultiCropDataset(
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=True,
    )

    # Get a set of transformed samples
    index, multi_crops = dataset[1]

    mean = [0.485, 0.456, 0.406]
    std = [0.228, 0.224, 0.225]

    # Display the transformed images
    fig, axes = plt.subplots(1, len(multi_crops), figsize=(15, 5))
    for i, crop in enumerate(multi_crops):
        print(crop.size)
        crop = crop.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        crop = crop * torch.tensor(std) + torch.tensor(mean)  # Denormalize
        crop = np.clip(crop.numpy(), 0, 1)  # Ensure values are within [0, 1]
        axes[i].imshow(crop)
        axes[i].axis('off')
    plt.show()
```
If your dataset returns a single tensor, the output from the DataLoader will be a tensor of shape (batch_size, *sample_shape).

If your dataset returns multiple tensors, the output from the DataLoader will be a tuple containing multiple tensors, each of shape (batch_size, *sub_sample_shape).

If your dataset returns a custom data structure, the output from the DataLoader will preserve that data structure, and its elements will be batched accordingly.

