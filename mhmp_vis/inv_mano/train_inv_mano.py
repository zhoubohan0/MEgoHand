import numpy as np
import torch, os, json, argparse
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter
from utils.rot import rot6d_to_aa, aa_to_rot6d, rot6d_to_rotmat, batch_rodrigues
from utils.util import seed_everything, vis_mano, mano_forward, normalize, denormalize


# architecture
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MiniPointNet(nn.Module):
    def __init__(self):
        super(MiniPointNet, self).__init__()
        # Shared MLP layers applied to each point (3D → 64 → 128 → 512)
        self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 512)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        # x: [B, 21, 3]
        B, N, _ = x.shape

        x = F.relu(self.bn1(self.fc1(x).transpose(1, 2))).transpose(1, 2)  # [B, 21, 64]
        x = F.relu(self.bn2(self.fc2(x).transpose(1, 2))).transpose(1, 2)  # [B, 21, 128]
        x = F.relu(self.bn3(self.fc3(x).transpose(1, 2))).transpose(1, 2)  # [B, 21, 512]

        # Max pooling across points to get global feature
        x = torch.max(x, dim=1)[0]  # [B, 512]
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels=512):
        super(ResBlock, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.LayerNorm(mid_channels),
            Swish(),
            nn.Linear(mid_channels, in_channels),
            nn.LayerNorm(in_channels),
        )

    def forward(self, x):
        return self.l1(x) + x

class InvMANOModel(nn.Module):
    def __init__(self):
        super(InvMANOModel, self).__init__()
        self.linera_input = nn.Sequential(
            nn.Flatten(),
            nn.Linear(21*3, 256),
            nn.LayerNorm(256),
            Swish()
        )
        self.resblock = nn.Sequential(*[ResBlock(256) for _ in range(16)])
        self.linera_output = nn.Linear(256, 15 * 6 + 10 + 6 + 3)
        
    def forward(self, x):
        x = self.linera_input(x)
        x = self.resblock(x)
        out = self.linera_output(x)
        return dict(
            theta=out[:, :90].view(-1, 15, 6),
            beta=out[:, 90:100],
            rot=out[:, 100:106].view(-1, 6),
            trans=out[:, 106:109]
        )
        
# dataset
class InvMANODataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        '''
        [
            'keypoints': (B, 21, 3), 
            'mano_params':
                - theta: (B, 15 * 3),
                - beta: (B, 10),
                - rot: (B, 3),
                - trans: (B, 3) 
        ]
        '''
        self.data = data_dict

    def __len__(self):
        return len(self.data['keypoints'])

    def __getitem__(self, idx):
        keypoints = torch.Tensor(self.data['keypoints'][idx:idx+1]).cuda().squeeze(0)
        theta = torch.Tensor(self.data['mano_params']['theta'][idx:idx+1]).cuda()
        beta = torch.Tensor(self.data['mano_params']['beta'][idx:idx+1]).cuda()
        rot = torch.Tensor(self.data['mano_params']['rot'][idx:idx+1]).cuda()
        trans = torch.Tensor(self.data['mano_params']['trans'][idx:idx+1]).cuda()
        
        # mano_params: rot3d -> rot6d
        mano_params = dict(
            theta=aa_to_rot6d(theta.reshape(-1,3)).squeeze(0),
            beta=beta.squeeze(0),
            rot=aa_to_rot6d(rot.reshape(-1, 3)).squeeze(0),
            trans=trans.squeeze(0),
        )
        return keypoints, mano_params
        
# trainer
class InvMANOTrainer(nn.Module):
    def __init__(self, **config):
        super(InvMANOTrainer, self).__init__()
        # config
        self.config = config
        # train model
        self.model = InvMANOModel()
        # dataset
        self.train_loader, self.eval_loader = None, None
        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        # loss
        self.pos_loss = nn.SmoothL1Loss() #nn.HuberLoss(delta=0.01)
        self.rot_loss = nn.SmoothL1Loss() #nn.HuberLoss(delta=0.01)
        # loss schedule
        self.loss_schedule = dict(
            shape=dict(
                theta_loss=0.4,
                beta_loss=0.6,
            ),
            pose = dict(
                rot_loss=0.3,
                trans_loss=0.3,
                keypoint_loss=0.4,
            ),
            all = dict(
                theta_loss=0.2,
                beta_loss=0.1,
                rot_loss=0.1,
                trans_loss=0.1,
                keypoint_loss=0.5,
            )
        )
        self.num_update = 0
        # logs
        self.log_dir = config['log_dir']
        # load checkpoints
        self.load(config['checkpoints'])
        

    def set_dataloader(self, train_loader, eval_loader):
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def load(self, ckpt_file):
        if os.path.exists(ckpt_file):
            self.model.load_state_dict(torch.load(ckpt_file))
            print(f"Checkpoint loaded from {ckpt_file}")

    def log(self, loss_dict, timestep, mode='Train'):
        for k, v in loss_dict.items():
            self.writer.add_scalar(f'{mode}/{k}', v, timestep)

    def update(self, inputs, targets, is_train=True):
        preds = self.model(inputs)
        # rot6d -> rot3d
        pred_keypoints = mano_forward(
            rot6d_to_aa(preds['theta']).reshape(-1, 45),
            preds['beta'], 
            rot6d_to_aa(preds['rot']).reshape(-1, 3),
            preds['trans'],
            self.configs['side'], 
        )[1]
        pred_keypoints_wo_trans = mano_forward(
            rot6d_to_aa(preds['theta']).reshape(-1, 45),
            preds['beta'], 
            rot6d_to_aa(preds['rot']).reshape(-1, 3),
            torch.zeros_like(preds['trans']),
            self.configs['side'], 
        )[1]
        pred_keypoints_wo_rot = mano_forward(
            rot6d_to_aa(preds['theta']).reshape(-1, 45),
            preds['beta'], 
            torch.zeros_like(preds['trans']),
            preds['trans'],
            self.configs['side'], 
        )[1]
        pred_keypoints_wo_transformation = mano_forward(
            rot6d_to_aa(preds['theta']).reshape(-1, 45),
            preds['beta'], 
            torch.zeros_like(preds['trans']),
            torch.zeros_like(preds['trans']),
            self.configs['side'], 
        )[1]
        # loss
        if self.config['nomalize_inputs']:
            inputs = denormalize(inputs, self.key_mean, self.key_std)
        targets_rot = rot6d_to_rotmat(targets['rot']).reshape(-1, 3, 3)
        targets_trans = targets['trans'].unsqueeze(1)
        loss_dict = dict(
            theta_loss=self.rot_loss(preds['theta'], targets['theta']),
            beta_loss=self.pos_loss(preds['beta'], targets['beta']),
            rot_loss=self.rot_loss(preds['rot'], targets['rot']),
            trans_loss=self.pos_loss(preds['trans'], targets['trans']),
            keypoint_loss=self.pos_loss(pred_keypoints, inputs) + \
                self.pos_loss(pred_keypoints_wo_trans, inputs - targets_trans) + \
                self.pos_loss(pred_keypoints_wo_rot, (inputs - targets_trans) @ targets_rot + targets_trans) + \
                self.pos_loss(pred_keypoints_wo_transformation, (inputs - targets_trans) @ targets_rot)
        )
        if is_train: # backpropagation
            if self.config['stage'] == 1:  # stage 1: prioritize shape
                loss_weight = self.loss_schedule['all'] if self.num_update % self.config['alter_freq'] == 0 else self.loss_schedule['shape']
            elif self.config['stage'] == 2:  # stage 2: prioritize pose
                loss_weight = self.loss_schedule['all'] if self.num_update % self.config['alter_freq'] == 0 else self.loss_schedule['pose']
            else:
                raise NotImplementedError
            loss_dict['total_loss'] = loss = sum([v * loss_dict[k] for k, v in loss_weight.items()]) 
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.num_update += 1
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        return loss_dict

    def conduct_training(self, epoch):
        seed_everything(42)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.model.train()
        pbar = tqdm(range(1,1+epoch), desc="Training InvMANO", ncols=100)
        for epoch in pbar:
            for ib, batch in enumerate(self.train_loader):
                # calculate loss
                loss_dict = self.update(*batch)
                pbar.set_postfix(loss=f"{loss_dict['total_loss']:.6f}")
                # record
                self.log(loss_dict, epoch * len(self.train_loader) + ib)
            # eval and save
            if epoch % self.config['eval_freq'] == 0:
                eval_loss_dict = self.conduct_evaluate()
                self.log(eval_loss_dict, epoch, mode='Eval')
                torch.save(self.model.state_dict(), f'{self.log_dir}/invmano_{epoch}.pth')
    
    def conduct_evaluate(self):
        self.model.eval()
        eval_loss_dict = {}
        with torch.no_grad():
            for batch in self.eval_loader:
                loss_dict = self.update(*batch, is_train=False)
                for k, v in loss_dict.items():
                    eval_loss_dict[k] = eval_loss_dict.get(k, 0) + v
        eval_avg_loss_dict = {k: v / len(self.eval_loader) for k, v in eval_loss_dict.items()}
        self.model.train()
        return eval_avg_loss_dict
            
    def inference(self, test_data):
        with torch.no_grad():
            pred = self.model(torch.Tensor(test_data).cuda())
            # rot6d -> rot3d
            return {
                'theta': rot6d_to_aa(pred['theta']).reshape(-1, 45),
                'beta': pred['beta'],
                'rot': rot6d_to_aa(pred['rot']).reshape(-1, 3),
                'trans': pred['trans'],
            }

def get_skeleton(skeleton_path='inv_mano/fpha_s1_close_milk_1_skeleton.txt'):
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1) / 1000
    reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])
    skeleton = skeleton[:, reorder_idx]
    return skeleton

def get_train_dataset(side):
    # Data in world coordinates
    train_data = dict(np.load('inv_mano/mano_pair.npz', allow_pickle=True))
    train_data = train_data[side][()]
    train_data['keypoints'] = np.asarray(train_data['keypoints'])
    train_data['mano_params'] = {k: np.asarray(v) for k, v in train_data['mano_params'].items()}
    '''
    train_data = {
        'keypoints': np.random.rand(10000, 21, 3),  
        'mano_params': {
            'theta': np.random.rand(10000, 15 * 3),
            'beta': np.random.rand(10000, 10),
            'rot': np.random.rand(10000, 3),
            'trans': np.random.rand(10000, 3)
        }
    }
    '''
    return train_data

def train_inv_mano():    
    parser = argparse.ArgumentParser(description='InvMANO Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--checkpoints', '-c', type=str, default='')
    parser.add_argument('--stage', '-s', type=int, default=0, choices=[1, 2], help='stage 1: prioritize shape; stage 2: prioritize pose')
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--eval_freq', type=int, default=20, help='frequency of evaluation and saving checkpoints')
    parser.add_argument('--alter_freq', type=int, default=5, help='frequency of changing loss weights')
    parser.add_argument('--nomalize_inputs', '-n', action='store_true', help='whether normalize keypoints')
    parser.add_argument('--side', type=str, default="side")
    args = parser.parse_args()
    args.log_dir = os.path.join('inv_mano/logs/', args.exp_name)
    
    # configuration
    configs = vars(args)
    print(f'Train configs: {configs}')
    trainer = InvMANOTrainer(**configs).cuda()
    if args.mode == 'train':  # train
        # Data in world coordinates
        train_data = get_train_dataset(configs['side'])
        # Statistics of keypoints
        if args.nomalize_inputs:  # normalize ['keypoints']
            train_data['keypoints'], key_mean, key_std = normalize(train_data['keypoints'])
            os.makedirs(args.log_dir, exist_ok=True)
            args.mean_std_file = os.path.join(args.log_dir, 'mano_pair_meanstd.json') 
            with open(args.mean_std_file, 'w') as f:
                json.dump(dict(mean=key_mean.tolist(), std=key_std.tolist()), f, indent=4)
            trainer.key_mean = torch.Tensor(key_mean).cuda()
            trainer.key_std = torch.Tensor(key_std).cuda()
        num_sample = len(train_data["keypoints"])
        print(f"center of keypoints (min, max): {train_data['keypoints'].min(axis=0).min(axis=0)}, {train_data['keypoints'].max(axis=0).max(axis=0)}")
        print(f"center of keypoints (mean, std): {train_data['keypoints'].mean(axis=0).mean(axis=0)}, {train_data['keypoints'].std(axis=0).std(axis=0)}")
        print(f'Train dataset has been loaded with {num_sample} samples.')
        # shuffle, then split train and eval data
        indices = np.arange(num_sample)
        np.random.shuffle(indices)
        num_train = int(0.9 * len(indices))
        train_indices, eval_indices = indices[:num_train], indices[num_train:]
        train_dataset = InvMANODataset({
            'keypoints': train_data['keypoints'][train_indices],
            'mano_params': {k: v[train_indices] for k, v in train_data['mano_params'].items()}
        })
        eval_dataset = InvMANODataset({
            'keypoints': train_data['keypoints'][eval_indices],
            'mano_params': {k: v[eval_indices] for k, v in train_data['mano_params'].items()}
        })
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8192, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8192, shuffle=False)
        trainer.set_dataloader(train_loader, eval_loader)
        trainer.conduct_training(epoch=args.epoch)
    else:
        '''    
        test_data = get_train_dataset()
        indices = np.random.choice(len(test_data['keypoints']), 32, replace=False)
        test_dataset = InvMANODataset({
            'keypoints': test_data['keypoints'][indices],
            'mano_params': {k: v[indices] for k, v in test_data['mano_params'].items()}
        })
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        for batch in test_loader:
            keypoints, mano_params = batch
            loss_dict = trainer.update(keypoints, mano_params, is_train=False)
            print(loss_dict)
            # original
            mano_params['theta'] = rot6d_to_aa(mano_params['theta']).reshape(-1, 45)
            mano_params['rot'] = rot6d_to_aa(mano_params['rot']).reshape(-1, 3)
            joint_ori = vis_mano(**mano_params,side=configs['side'])
            # overwrite pose = 0
            from utils.rot import batch_aa2rotmat
            rotmat_back = batch_aa2rotmat(mano_params['rot'])
            trans_back = mano_params['trans']
            mano_params['rot'] = torch.zeros_like(mano_params['rot'])
            # mano_params['trans'] = torch.zeros_like(mano_params['trans'])
            joint_nor = vis_mano(**mano_params, side=configs['side'])
            breakpoint()
            assert np.allclose(joint_ori, joint_nor @ rotmat_back.transpose(1, 2)[0].cpu().numpy() + trans_back.cpu().numpy(), atol=1e-5)
            pred_mano_param = trainer.inference(keypoints)
        '''
        
        # test_data = get_skeleton()
        test_data = get_train_dataset(configs['side'])['keypoints']
        if not args.checkpoints:
            args.checkpoints = natsorted(glob(os.path.join(args.log_dir, '*.pth')))[-1]
        trainer.load(args.checkpoints)
        if args.nomalize_inputs:
            with open(args.mean_std_file, 'r') as f:
                mean_std_dict = json.load(f)
                key_mean, key_std = np.array(mean_std_dict['mean']), np.array(mean_std_dict['std'])
                test_data = (test_data - key_mean) / key_std
        for t in np.random.choice(len(test_data), 32, replace=False):
            pred_mano_param = trainer.inference(test_data[t:t+1])
            vis_mano(**pred_mano_param, side=configs['side'],ref_keypoints=torch.Tensor(test_data[t:t+1]).cuda(),)


# ----------------------------------------------------------------------

class DirectInvMANOTrainer:
    def __init__(self):
        # loss
        self.pos_loss = nn.SmoothL1Loss()
        # log
        self.log_dir = 'inv_mano/logs/direct'

    def conduct_training(self, mano_keypoints, side, save_file='', epoch=2000, loss_thre=9e-5, is_writer=True):
        '''
        mano_keypoints: (N, 21, 3) torch tensor
        '''
        num_sample = len(mano_keypoints)
        # input
        if getattr(self, 'mano_param', None) is None:
            self.batch_theta = nn.Parameter(0.01 * torch.randn(num_sample, 15 * 6, requires_grad=True).cuda())
            self.beta = nn.Parameter(0.01 * torch.randn(1, 10).cuda())
            self.batch_rot = nn.Parameter(0.01 * torch.randn(num_sample, 6, requires_grad=True).cuda())
            self.batch_trans = nn.Parameter(0.01 * torch.randn(num_sample, 3, requires_grad=True).cuda())
        else:
            self.batch_theta = nn.Parameter(self.batch_theta[:1].repeat(num_sample, 1).cuda(), requires_grad=True)
            self.batch_rot = nn.Parameter(self.batch_rot[:1].repeat(num_sample, 1).cuda(), requires_grad=True)
            self.batch_trans = nn.Parameter(self.batch_trans[:1].repeat(num_sample, 1).cuda(), requires_grad=True)
        # optimizer
        self.optimizer = torch.optim.Adam([self.batch_theta, self.beta, self.batch_rot, self.batch_trans], lr=5e-3, weight_decay=1e-4, betas=(0.9, 0.99), eps=1e-5)
        # log
        if is_writer:
            writer = SummaryWriter(log_dir=self.log_dir)
        seed_everything(42)
        pbar = tqdm(range(epoch), total=epoch, desc='Training InvMANO', ncols=100)
        for iter in pbar:
            mano_param = self.inference()
            mano_theta, mano_beta, wrist_rot, wrist_trans = mano_param['theta'], mano_param['beta'], mano_param['rot'], mano_param['trans']
            
            wrist_rot9d = batch_rodrigues(wrist_rot)
            pred_keypoints_w_transform = mano_forward(mano_theta, mano_beta, wrist_rot, wrist_trans, side)[1]
            pred_keypoints_w_rot = mano_forward(mano_theta, mano_beta, wrist_rot, torch.zeros_like(wrist_trans), side)[1]
            pred_keypoints_w_trans = mano_forward(mano_theta, mano_beta, torch.zeros_like(wrist_rot), wrist_trans, side)[1]
            pred_keypoints_decentralize = mano_forward(mano_theta, mano_beta, torch.zeros_like(wrist_rot), torch.zeros_like(wrist_trans), side)[1]
        
            loss_dict = dict(
                l1 = self.pos_loss(pred_keypoints_w_transform, mano_keypoints),
                l2 = self.pos_loss(pred_keypoints_w_rot, mano_keypoints - wrist_trans[:, None,:]),
                l3 = self.pos_loss(pred_keypoints_w_trans, (mano_keypoints - wrist_trans[:, None,:]) @ wrist_rot9d + wrist_trans[:, None,:]),
                l4 = self.pos_loss(pred_keypoints_decentralize, (mano_keypoints - wrist_trans[:, None,:]) @ wrist_rot9d),
                l5 = self.pos_loss(pred_keypoints_w_trans, pred_keypoints_w_rot @ wrist_rot9d + wrist_trans[:, None,:]),
                l6 = self.pos_loss(pred_keypoints_decentralize, pred_keypoints_w_rot @ wrist_rot9d),
                l7 = self.pos_loss(pred_keypoints_w_trans, pred_keypoints_decentralize + wrist_trans[:, None,:]),
            )
            loss = sum(loss_dict.values())
            if loss.item() < loss_thre:
                break
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix(loss=f'{loss.item():.6f}')
            if is_writer:# log
                for k, v in loss_dict.items():
                    writer.add_scalar('Loss/' + k, v, iter)
        # save checkpoint
        mano_param = self.inference()
        if save_file:
            with open(save_file, 'w') as f:
                json.dump({k: v.tolist() for k, v in mano_param.items()}, f, indent=4)
        
        return {k: v.detach() for k, v in mano_param.items()}
    
    def inference(self):
        return dict(
            theta=rot6d_to_aa(self.batch_theta.reshape(-1,6)).reshape(-1, 45),
            beta=self.beta,
            rot=rot6d_to_aa(self.batch_rot).reshape(-1, 3),
            trans=self.batch_trans,
        )
    
    def visualize(self, mano_param_dict, side, mano_keypoints=None):
        '''
        mano_param_dict: dict(
            theta: (N, 45) torch tensor,
            beta: (1, 10) torch tensor,
            rot: (N, 3) torch tensor,
            trans: (N, 3) torch tensor
        )
        mano_keypoints: (N, 21, 3) torch tensor
        '''
        for t in range(len(mano_param_dict['theta'])):
            vis_mano(
                theta=mano_param_dict['theta'][t:t+1],
                beta=mano_param_dict['beta'],
                rot=mano_param_dict['rot'][t:t+1],
                trans=mano_param_dict['trans'][t:t+1],
                side=side, 
                ref_keypoints=mano_keypoints[t:t+1]
            )

def train_direct_inv_mano():
    # data
    src_file = 'inv_mano/fpha_s1_close_milk_1_skeleton.txt'
    mano_keypoints = torch.Tensor(get_skeleton(src_file)).cuda().detach()
    trainer = DirectInvMANOTrainer()
    save_file = src_file.replace('inv_mano', trainer.log_dir).replace('skeleton.txt', 'mano.json')
    if not os.path.exists(save_file):  # training
        mano_param_dict = trainer.conduct_training(mano_keypoints, 'right', save_file)
    else:
        with open(save_file, 'r') as f:
            mano_param_dict = json.load(f)
        mano_param_dict = {k: torch.Tensor(v).cuda() for k, v in mano_param_dict.items()}
    # visualize
    trainer.visualize(mano_param_dict, mano_keypoints)


if __name__ == "__main__":
    train_direct_inv_mano()