from random import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 파일 리드
with open("csv파일/bream_length.csv") as file_name:
    bream_length = np.loadtxt(file_name, delimiter=",")
#print(bream_length.shape)
#print(bream_length)

with open("csv파일/bream_weight.csv") as file_name:
    bream_weight = np.loadtxt(file_name, delimiter=",")
#print(bream_weight.shape)
#print(bream_weight)

with open("csv파일/smelt_length.csv") as file_name:
    smelt_length = np.loadtxt(file_name, delimiter=",")
#print(smelt_length.shape)
#print(smelt_length)

with open("csv파일/smelt_weight.csv") as file_name:
    smelt_weight = np.loadtxt(file_name, delimiter=",")
#print(smelt_weight.shape)
#print(smelt_weight)

# 시각화
# bream = plt.scatter(bream_length, bream_weight)
# smelt = plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
#plt.show()

# 배열 합치기
bream = bream_length + bream_weight
smelt = smelt_length + smelt_weight

print(bream.shape)
print(smelt.shape)
# print(bream)
# print(smelt)



# 2차원 배열 합치기
bream_data = np.column_stack((bream_length, bream_weight))
# print(bream_data.shape)

smelt_data = np.column_stack((smelt_length, smelt_weight))
# print(smelt_data.shape)

# 1차원 배열 합치기
fish_data_d1= np.concatenate((bream_data, smelt_data), axis=0)
# print(fish_data_d1)
# print(fish_data_d1.shape)

# 학습
fish_target = np.array([1]*35 + [0]*14)
print(fish_target)

np.random.seed(42)
index = np.arange(49)
shuffle_data = np.random.shuffle(index)
print(index)

train_input = fish_data_d1[index[:35]]
train_target = fish_target[index[:35]]

test_input = fish_data_d1[index[35:]]
test_target = fish_target[index[:35]]

print(fish_data_d1[13], train_input[0])

# 훈련 데이터, 도미와 빙어 점 찍기
# 테스트 데이터, 도미와 빙어 점 찍기
plt.scatter(train_input[:, 0], train_input[:, 1]) 
plt.scatter(test_input[:, 0], test_input[:, 1]) 
plt.xlabel('length')
plt.ylabel('weight')
plt.show()