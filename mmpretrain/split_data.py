import os
import random
import shutil

random.seed(42)  # 固定随机种子，保证每次划分一致

# ===== 1. 路径设置 =====
# 现在已有的数据根目录
src_root = '/root/autodl-tmp/mmpretrain/mmpretrain/dogs_vs_cats'

# 划分后要生成的新数据集根目录
dst_root = '/root/autodl-tmp/mmpretrain/mmpretrain/dogs_vs_cats_split'

classes = ['cat', 'dog']  # 你的类别文件夹名就是 cat / dog

# 划分比例：70% 训练，15% 验证，15% 测试
train_ratio = 0.7
val_ratio = 0.15  # 剩下的自动当 test

# ===== 2. 创建目标目录结构 =====
splits = ['train', 'val', 'test']
for split in splits:
    for cls in classes:
        out_dir = os.path.join(dst_root, split, cls)
        os.makedirs(out_dir, exist_ok=True)

# ===== 3. 按类划分并拷贝文件 =====
for cls in classes:
    src_dir = os.path.join(src_root, cls)
    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files   = files[n_train:n_train + n_val]
    test_files  = files[n_train + n_val:]

    print(f'{cls}: total={n}, train={len(train_files)}, '
          f'val={len(val_files)}, test={len(test_files)}')

    def move_files(file_list, split):
        for fname in file_list:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_root, split, cls, fname)
            # 如果你想保留原始 dogs_vs_cats，可以换成 shutil.copy2
            shutil.copy2(src_path, dst_path)

    move_files(train_files, 'train')
    move_files(val_files,   'val')
    move_files(test_files,  'test')

print('Done! New dataset at:', dst_root)
