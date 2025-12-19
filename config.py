# -*- coding: utf-8 -*-
import string

# 字母表
letters = string.ascii_letters + " ,.;'"
n_letters = len(letters)  # 57

# 18个国家（类别）
categorys = [
    'Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish',
    'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch',
    'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German'
]
n_categories = len(categorys)

# 训练超参数
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 3          #3轮以上才能看出效果，1轮太少
BATCH_SIZE = 1
DATA_PATH = "./data/name_classfication.txt"










