import os
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from copy import deepcopy
import data_util
import matplotlib.pyplot as plt


class NovelViewTriplets():
    def __init__(self,
                 root_dir,
                 img_size,
                 sampling_pattern):
        super().__init__()

        self.img_size = img_size
        
        images_dir = [os.path.join(root_dir, o) for o in os.listdir(root_dir) 
                    if os.path.isdir(os.path.join(root_dir,o))]
        
        self.color_dir = {}
        self.pose_dir = {}
        self.all_color = {}
        self.all_poses = {}
        self.all_views = {}
        self.nn_idcs = {}

        for image_dir in images_dir:
            self.color_dir[image_dir] = os.path.join(image_dir, 'rgb')
            self.pose_dir[image_dir] = os.path.join(image_dir, 'pose')

            if not os.path.isdir(self.color_dir[image_dir]):
                print("Error! root dir is wrong")
                return

            self.all_color[image_dir] = sorted(data_util.glob_imgs(self.color_dir[image_dir]))
            self.all_poses[image_dir] = sorted(glob(os.path.join(self.pose_dir[image_dir], '*.txt')))

            # Subsample the trajectory for training / test set split as well as the result matrix
            file_lists = [self.all_color[image_dir], self.all_poses[image_dir]]

            if sampling_pattern != 'all':
                if sampling_pattern.split('_')[0] == 'skip':
                    skip_val = int(sampling_pattern.split('_')[-1])

                    for i in range(len(file_lists)):
                        dummy_list = deepcopy(file_lists[i])
                        file_lists[i].clear()
                        file_lists[i].extend(dummy_list[::skip_val + 1])
                else:
                    print("Unknown sampling pattern!")
                    return None

        # Buffer files
        print("Buffering files...")
        for image_dir in images_dir:
            self.all_views[image_dir] = []
            for i in range(len(self.all_color[image_dir])):
                if not i % 10:
                    print(i)
                self.all_views[image_dir].append(self.read_view_tuple(image_dir, i))

            # Calculate the ranking of nearest neigbors
            # print(len(data_util.get_nn_ranking([data_util.load_pose(pose) for pose in self.all_poses[image_dir]])[0]))
            self.nn_idcs[image_dir], _ = data_util.get_nn_ranking([data_util.load_pose(pose) for pose in self.all_poses[image_dir]])

        print("*" * 100)
        print("Sampling pattern ", sampling_pattern)
        print("Image size ", self.img_size)
        print("*" * 100)
        
        
        sizes = []
        self.images_dir = images_dir
        curr = 0
        for image_dir in images_dir:
            curr += len(self.all_color[image_dir])
            sizes.append(curr)
        self.sizes = np.array(sizes)

    def load_rgb(self, path):
        img = data_util.load_img(path, square_crop=True, downsampling_order=1, target_size=self.img_size)
        img = img[:, :, :3].astype(np.float32) / 255. - 0.5
        img = img.transpose(2,0,1)
        return img

    def read_view_tuple(self, image_dir, idx):
        gt_rgb = self.load_rgb(self.all_color[image_dir][idx])
        pose = data_util.load_pose(self.all_poses[image_dir][idx])

        this_view = {'gt_rgb': torch.from_numpy(gt_rgb),
                     'pose': torch.from_numpy(pose)}
        return this_view

    def idx2imgdir(self, idx):
        img_dir_idx = np.argmax(self.sizes>idx)
        if img_dir_idx == 0:
            return self.images_dir[img_dir_idx], idx
        else:
            return self.images_dir[img_dir_idx], idx - self.sizes[img_dir_idx-1]

    def __len__(self):
        a = [len(self.all_color[i]) for i in self.all_color]
        return sum(a)

    def __getitem__(self, idx):
        trgt_views = []

        # Read one target pose and its nearest neighbor
        image_dir, iidx = self.idx2imgdir(idx)
        trgt_views.append(self.all_views[image_dir][iidx])
        nearest_view = self.all_views[image_dir][self.nn_idcs[image_dir][iidx][-np.random.randint(low=1, high=5)]]

        # The second target pose is a random one
        image_dir = np.random.choice(list(self.all_views.keys()))
        trgt_views.append(self.all_views[image_dir][np.random.choice(len(self.all_views[image_dir]))])

        return trgt_views, nearest_view, image_dir


class TestDataset():
    def __init__(self,
                 pose_dir):
        super().__init__()

        all_pose_paths = sorted(glob(os.path.join(pose_dir, '*.txt')))
        self.all_poses = [torch.from_numpy(data_util.load_pose(path)) for path in all_pose_paths]

    def __len__(self):
        return len(self.all_poses)

    def __getitem__(self, idx):
        return self.all_poses[idx]
