# -*- coding: utf-8 -*-
import json
import torch
import matplotlib.pyplot as plt
from config import letters, categorys, DEVICE
from models import RNN, LSTM, GRU

def line_to_tensor(line):
    tensor = torch.zeros(len(line), len(letters))
    for i, char in enumerate(line):
        tensor[i][letters.find(char)] = 1
    return tensor

def predict(name, model_class, weight_path):
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        x = line_to_tensor(name).to(DEVICE)
        if model_class == LSTM:
            h, c = model.init_hidden()
            out, _, _ = model(x, h, c)#只取我想要的值，其余的我不care
        else:
            h = model.init_hidden()
            out, _ = model(x, h)

        values, indices = torch.topk(out, k=3, dim=-1)
        print(f"\n预测姓名: {name}")
        for v, idx in zip(values[0], indices[0]):
            print(f"  {categorys[idx]:15}  概率: {torch.exp(v):.4f}")

def plot_results():
    names = ["rnn", "lstm", "gru"]
    colors = ["blue", "red", "orange"]
    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 3, 1)
    for name, color in zip(names, colors):
        with open(f"./json/ai_{name}.json", encoding="utf-8") as f:
            data = json.load(f)
        plt.plot(data["avg_loss"], label=name.upper(), color=color)
    plt.title("Loss Curve")
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 2)
    for name, color in zip(names, colors):
        with open(f"./json/ai_{name}.json", encoding="utf-8") as f:
            data = json.load(f)
        plt.plot(data["avg_acc"], label=name.upper(), color=color)
    plt.title("Accuracy Curve")
    plt.legend()

    # Time
    plt.subplot(1, 3, 3)
    times = []
    for name in names:
        with open(f"./json/ai_{name}.json", encoding="utf-8") as f:
            data = json.load(f)
        times.append(data["all_time"])
    plt.bar(names, times, color=colors)
    plt.title("Training Time (seconds)")

    plt.tight_layout()
    plt.savefig("./phone/comparison.png")

    plt.show()

