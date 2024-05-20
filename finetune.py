from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch
import os
import csv

def main(batch_size=32, learning_rate=0.001,device="cuda"):


    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=os.path.join('D:\\PycharmProjects\\Image_Cls\\CUB_200_2011', 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join('D:\\PycharmProjects\\Image_Cls\\CUB_200_2011', 'test'), transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 加载预训练的模型
    model = models.resnet50(pretrained=False)
    '''
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False
    '''
    # 替换最后的全连接层以匹配类别数
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 200)
    model = model.to(device)  # 确保在修改模型后发送到设备
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)

    # 初始化 TensorBoard
    writer = SummaryWriter(f'runs/trainedall_ResNet50_bs{batch_size}_lr{learning_rate}')



    # 训练模型并保存最佳模型权重
    best_acc = 0.0
    results = []
    for epoch in range(200):
        train_loss, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'trained_model_weights_bs{batch_size}_lr{learning_rate}.pth')

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 评估测试集性能
    test_acc = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.4f}')

    return {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'best_val_acc': best_acc,
        'test_acc': test_acc
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 确保输入和标签在同一设备
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    val_acc = running_corrects.double() / len(val_loader.dataset)
    return train_loss, val_acc

def evaluate_model(model, test_loader, device):
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    return running_corrects.double() / len(test_loader.dataset)


def run_experiments():
    configurations = [
        #{'batch_size': 64, 'learning_rate': 0.002},
        {'batch_size': 16, 'learning_rate': 0.01},
       #{'batch_size': 32, 'learning_rate': 0.0015}
    ]
    # 确定是否有CUDA支持的GPU可用，否则继续使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    results = []
    for config in configurations:
        result = main(batch_size=config['batch_size'], learning_rate=config['learning_rate'],device=device)
        results.append(result)

    # 将结果写入 CSV 文件
    #write_results_to_csv(results)


def write_results_to_csv(results):
    keys = results[0].keys()
    with open('training_results_ft.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


if __name__ == '__main__':
    run_experiments()
