import os
import shutil
import pandas as pd

# 数据集根目录
dataset_path = 'D:\\PycharmProjects\\Image_Cls\\CUB_200_2011'
# 图片文件夹路径
images_path = os.path.join(dataset_path, 'images')

# 读取数据集信息
classes_df = pd.read_csv(os.path.join(dataset_path, 'classes.txt'), sep=' ', header=None,
                         names=['class_id', 'class_name'])
image_labels_df = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'), sep=' ', header=None,
                              names=['image_id', 'class_id'])
train_test_split_df = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'), sep=' ', header=None,
                                  names=['image_id', 'is_train'])
images_df = pd.read_csv(os.path.join(dataset_path, 'images.txt'), sep=' ', header=None, names=['image_id', 'file_path'])

# 准备目标文件夹
for directory in ['train', 'test']:
    directory_path = os.path.join(dataset_path, directory)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# 开始复制图片
for _, image_row in images_df.iterrows():
    image_id = image_row['image_id']
    file_path = image_row['file_path']
    class_id = image_labels_df.loc[image_labels_df['image_id'] == image_id, 'class_id'].values[0]
    is_train = train_test_split_df.loc[train_test_split_df['image_id'] == image_id, 'is_train'].values[0]

    class_name = classes_df.loc[classes_df['class_id'] == class_id, 'class_name'].values[0].replace(' ', '_')
    subdir = 'train' if is_train else 'test'
    class_path = os.path.join(dataset_path, subdir, class_name)

    if not os.path.exists(class_path):
        os.makedirs(class_path)

    # 构建源文件和目标文件路径
    source_path = os.path.join(images_path, file_path)
    target_path = os.path.join(class_path, os.path.basename(file_path))

    # 检查源文件是否存在，并复制到目标路径
    if os.path.isfile(source_path):
        shutil.copy(source_path, target_path)
    else:
        print(f'File not found: {source_path}')
