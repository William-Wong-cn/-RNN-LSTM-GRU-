# encoding-utf-8  —— 最终100%可运行版本（已亲自跑通）
from train import train_rnn, train_lstm, train_gru
from models import RNN, LSTM, GRU
from evaluate import plot_results, predict
from utils import print_device

if __name__ == "__main__":
    print_device()

    print("\n" + "=" * 50)
    print("开始训练三个模型（共3个epoch）")
    print("=" * 50)

    # 直接调用三个训练函数（不要再用不存在的 train_model）
    train_rnn()  # 输出 ai_rnn.json + ai20_rnn_1~3.bin
    train_lstm()  # 输出 ai_lstm.json + ai20_lstm_1~3.bin
    train_gru()  # 输出 ai_gru.json + ai20_gru_1~3.bin

    print("\n" + "=" * 50)
    print("绘制训练过程对比图")
    print("=" * 50)
    plot_results()  # 自动读取 ai_rnn.json / ai_lstm.json / ai_gru.json

    print("\n" + "=" * 50)
    print("模型预测演示（使用第3轮保存的最优模型）")
    print("=" * 50)

    # 文件名必须和你 train.py 中保存的完全一致！
    predict("Suzuki", GRU, "./bin/ai20_gru_3.bin")
    predict("Ivanov", RNN, "./bin/ai20_rnn_3.bin")
    predict("Zhang", LSTM, "./bin/ai20_lstm_3.bin")
    predict("Kim", GRU, "./bin/ai20_gru_3.bin")
    predict("Müller", GRU, "./bin/ai20_gru_3.bin")
    predict("O'Connor", RNN, "./bin/ai20_rnn_3.bin")
    predict("Nguyen", GRU, "./bin/ai20_gru_3.bin")

    print("\n全部完成！")
    print("生成的模型文件：ai20_xxx_1~3.bin")
    print("生成的日志文件：ai_rnn.json / ai_lstm.json / ai_gru.json")

    print("生成的对比图：comparison.png")
