from gr00t.experiment.data_config import DATA_CONFIG_MAP
from mhmp_vis.dataset_loader import load_cascaded_dataset

data_config = DATA_CONFIG_MAP['hand_motion']
data_storage_kwargs = dict(window_size=16, side='right', is_render_depth=False)
# split whole train & evaluation dataset
test_dataset = load_cascaded_dataset(
    dataset_path='/media/ps/zhan/DATASET/',
    dataset_list='fpha'.split(','),
    split_portion=0.99,
    data_config=data_config,
    data_storage_kwargs=data_storage_kwargs,
    test=True,
    debug=False,
)
breakpoint()
# load partial testing dataset
train_dataset, eval_dataset = load_cascaded_dataset(
    dataset_path='/media/ps/zhan/DATASET/',
    dataset_list='fpha'.split(','),
    split_portion=0.95,
    data_config=data_config,
    data_storage_kwargs=data_storage_kwargs,
    test=False,
    debug=False,
)