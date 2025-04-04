import numpy as np
import importlib
import cv2
from torch.utils.data import Dataset
from dataset_loader import *


class CascadedDataset(Dataset):
    def __init__(self, dataset_list):
        """
        Cascade multiple MetaDataset instances specifically for training.
        """
        super().__init__()
        self.dataset_list = dataset_list
        self.lengths = [len(dataset) for dataset in dataset_list]
        self.cumsum_length_list = [0] + list(np.cumsum(self.lengths))

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx) -> dict:
        try:
            dataset_id = np.searchsorted(self.cumsum_length_list, idx+1) - 1
            sample_idx = idx - self.cumsum_length_list[dataset_id]
            dataset = self.dataset_list[dataset_id]
            sample = dataset[sample_idx]
            # resize image to 256,256
            sample['video.ego_view'] = cv2.resize(sample['video.ego_view'], (256, 256), interpolation=cv2.INTER_LINEAR)[None, ...]
            return sample
        except IndexError:
            error_log = ""
            if dataset_id >= len(self.dataset_list):
                error_log += f"Dataset ID {dataset_id} out of range because total datasets: {len(self.dataset_list)}.\n"
            if sample_idx >= len(dataset):
                error_log += f"Sample index {sample_idx} out of range because dataset length: {len(dataset)}.\n"
            print(error_log)
            self.__getitem__(np.random.randint(0, len(self)))


class CascadedDatasetWithTransform(CascadedDataset):
    def __init__(self, dataset_list, transform):
        super().__init__(dataset_list)
        self.transform = transform

    def __getitem__(self, idx) -> dict:
        return self.transform(super().__getitem__(idx))

if __name__ == "__main__":
    dataset_name_list = ["fpha", "hoi4d", ]  # "arctic" 
    dataset_list = []
    data_storage_kwargs = dict(window_size=2, side='right', is_render_depth=False)
    # Concatenate datasets
    for each in dataset_name_list:
        dataset_name = each.upper()
        dataset = eval(f"{dataset_name}Dataset")()
        dataset.initialize_data_storage(eval(f"{dataset_name}Storage"), data_storage_kwargs)
        dataset_list.append(dataset)
    cascaded_dataset = CascadedDataset(dataset_list)
    print(len(cascaded_dataset), cascaded_dataset.cumsum_length_list)
    for each in dataset_list:
        print(f"{each.__class__.__name__}: {len(each)}, {each.data_storage.cumsum_length_list}")
    # examine samples
    for i in range(0,len(cascaded_dataset),100):
        sample = cascaded_dataset[i]