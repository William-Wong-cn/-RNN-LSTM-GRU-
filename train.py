#encoding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入模块化文件
from config import DEVICE, LEARNING_RATE, EPOCHS
from data_utils import read_names, NameDataset
from models import RNN, LSTM, GRU
# ====================== 训练 RNN ======================
def train_rnn():
    names, labels = read_names()
    dataset = NameDataset(names, labels)

    model = RNN().to(DEVICE)                    # 直接用 RNN，不用传参数
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    total_iter_num = 0
    total_loss = 0.0
    total_acc_num = 0
    total_loss_list = []
    total_acc_list = []

    print("开始训练 RNN...")
    for epoch_idx in range(EPOCHS):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{EPOCHS}"):
            x = x[0].to(DEVICE)      # [seq_len, 57]
            y = y.to(DEVICE)

            hidden = model.init_hidden()                   # 不再传 device
            output, _ = model(x, hidden)

            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iter_num += 1
            total_loss += loss.item()
            if output.argmax(dim=1).item() == y.item():
                total_acc_num += 1

            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss / total_iter_num)
                total_acc_list.append(total_acc_num / total_iter_num)

            if total_iter_num % 2000 == 0:
                temp_time = int(time.time() - start_time)
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f'
                      % (epoch_idx+1, total_loss/total_iter_num, temp_time, total_acc_num/total_iter_num))

        # 每轮保存一次，和你原来一模一样
        torch.save(model.state_dict(), f'./bin/ai20_rnn_{epoch_idx+1}.bin')

    total_time = int(time.time() - start_time)
    print('训练总耗时：', total_time)

    result = {"avg_loss": total_loss_list, "all_time": total_time, "avg_acc": total_acc_list}
    with open('./json/ai_rnn.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    return total_loss_list, total_time, total_acc_list


# ====================== 训练 LSTM ======================
def train_lstm():
    names, labels = read_names()
    dataset = NameDataset(names, labels)

    model = LSTM().to(DEVICE)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    total_iter_num = 0
    total_loss = 0.0
    total_acc_num = 0
    total_loss_list = []
    total_acc_list = []

    print("开始训练 LSTM...")
    for epoch_idx in range(EPOCHS):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{EPOCHS}"):
            x = x[0].to(DEVICE)
            y = y.to(DEVICE)

            hidden, cell = model.init_hidden()            # 返回两个
            output, _, _ = model(x, hidden, cell)

            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iter_num += 1
            total_loss += loss.item()
            if output.argmax(dim=1).item() == y.item():
                total_acc_num += 1

            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss / total_iter_num)
                total_acc_list.append(total_acc_num / total_iter_num)

            if total_iter_num % 2000 == 0:
                temp_time = int(time.time() - start_time)
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f'
                      % (epoch_idx+1, total_loss/total_iter_num, temp_time, total_acc_num/total_iter_num))

        torch.save(model.state_dict(), f'./bin/ai20_lstm_{epoch_idx+1}.bin')

    total_time = int(time.time() - start_time)
    print('训练总耗时：', total_time)

    result = {"avg_loss": total_loss_list, "all_time": total_time, "avg_acc": total_acc_list}
    with open('./json/ai_lstm.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    return total_loss_list, total_time, total_acc_list


# ====================== 训练 GRU ======================
def train_gru():
    names, labels = read_names()
    dataset = NameDataset(names, labels)

    model = GRU().to(DEVICE)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    total_iter_num = 0
    total_loss = 0.0
    total_acc_num = 0
    total_loss_list = []
    total_acc_list = []

    print("开始训练 GRU...")
    for epoch_idx in range(EPOCHS):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{EPOCHS}"):
            x = x[0].to(DEVICE)
            y = y.to(DEVICE)

            hidden = model.init_hidden()
            output, _ = model(x, hidden)

            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iter_num += 1
            total_loss += loss.item()
            if output.argmax(dim=1).item() == y.item():
                total_acc_num += 1

            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss / total_iter_num)
                total_acc_list.append(total_acc_num / total_iter_num)

            if total_iter_num % 2000 == 0:
                temp_time = int(time.time() - start_time)
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f'
                      % (epoch_idx+1, total_loss/total_iter_num, temp_time, total_acc_num/total_iter_num))

        torch.save(model.state_dict(), f'./bin/ai20_gru_{epoch_idx+1}.bin')

    total_time = int(time.time() - start_time)
    print('训练总耗时：', total_time)

    result = {"avg_loss": total_loss_list, "all_time": total_time, "avg_acc": total_acc_list}
    with open('./json/ai_gru.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    return total_loss_list, total_time, total_acc_list




