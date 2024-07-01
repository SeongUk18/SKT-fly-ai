#pytorch를 이용한 간단한 Fashion-MNIST Datatset classifier 구현 



import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


import matplotlib.pyplot as plt
import tqdm
import numpy as np
import random

#1. hyper parameter 설정

batch_size = 64 # batch size 정의
learning_rate = 1e-3 # learning rate 정의
weight_decay = 1e-3 # weight decay 정의
dropout_rate = 0.2 # dropout rate 정의
epochs = 50 # epoch 횟수 정의


#2. randomness 설정
random_seed = 100 # random seed 값 고정
torch.manual_seed(random_seed) # pytorch 사용 시 randomness 설정
torch.cuda.manual_seed(random_seed) # GPU 사용하는 경우 randomness 설정
torch.cuda.manual_seed_all(random_seed) # multi-GPU 사용하는 경우 randomness 설정
torch.backends.cudnn.deterministic = True # cudnn에서 randomness 설정
torch.backends.cudnn.benchmark = False # cudnn에서 randomness 설정
np.random.seed(random_seed) #python의 numpy에서 제공되는 random 함수 고정
random.seed(random_seed) #python에서 제공되는 random 함수 고정

#3. PyTorch는 TorchText, TorchVision 및 TorchAudio 와 같이 도메인 특화 라이브러리를
# 데이터셋과 함께 제공하고 있습니다. 이 튜토리얼에서는 TorchVision 데이터셋을 사용하도록
# 하겠습니다. Torchvision.datasets 모듈은 CIFAR, COCO 등과 같은 다양한 실제 영상(vision)
# 데이터에 대한 Dataset를 포함하고 있습니다. 이 튜토리얼에서는 
# FasionMNIST 데이터셋을 사용합니다. 모든 TorchVision Dataset 은 샘플과 정답을 각각 
# 변경하기 위한 transform 과 target_transform 의 두 인자를 포함합니다.

# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Dataset 을 DataLoader 의 인자로 전달합니다. 
# 이는 데이터셋을 순회 가능한 객체(iterable)로 감싸고, 자동화된 배치(batch),
# 샘플링(sampling), 섞기(shuffle) 및 다중 프로세스로 데이터 불러오기(multiprocess data loading)를
# 지원합니다. 여기서는 배치 크기(batch size)를 64로 정의합니다. 즉, 데이터로더(dataloader) 객체의 
# 각 요소는 64개의 특징(feature)과 정답(label)을 묶음(batch)으로 반환합니다.



# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#4. 모델 만들기
#PyTorch에서 신경망 모델은 nn.Module 을 상속받는 클래스(class)를 생성하여 정의합니다.
# __init__ 함수에서 신경망의 계층(layer)들을 정의하고 forward 함수에서 신경망에 데이터를
# 어떻게 전달할지 지정합니다. 가능한 경우 GPU 또는 MPS로 신경망을 이동시켜
# 연산을 가속(accelerate)합니다.

# 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Convolutional Neural Network 모델을 정의합니다.
class CnnNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)
        self.bn2d1 = nn.BatchNorm2d(num_features=256)
        
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn2d2 = nn.BatchNorm2d(num_features=128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.bn2d3 = nn.BatchNorm2d(num_features=32)

        self.linear1 = nn.Linear(in_features=32*7*7, out_features=256)
        self.bn1d1 = nn.BatchNorm1d(num_features=256)
        
        self.linear2 = nn.Linear(in_features=256,out_features=10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.dropout2d = nn.Dropout2d(dropout_rate)
        self.dropout1d = nn.Dropout1d(dropout_rate)
    
        # 학습 가능한 parameter 초기화 하는 방법
        # 모델의 모듈을 차례대로 불러옵니다.
        for m in self.modules():
            #모듈이 nn.Conv2d인 경우
            if isinstance(m, nn.Conv2d):
                '''
                # 작은 숫자로 초기화하는 방법
                # 가중치를 평균 0, 편차 0.02로 초기화합니다.
                # bias를 0으로 초기화합니다.
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                
                # Xavier Initialization
                # 모듈의 가중치를 xavier normal로 초기화합니다.
                # bias를 0으로 초기화합니다.
                #nn.init.xavier_normal_(m.weight.data)
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                '''
                # Kaming Initialization
                # 모듈의 가중치를 kaming he normal로 초기화합니다.
                # bias를 0으로 초기화합니다.
                #nn.init.kaiming_normal_(m.weight.data)
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            
            # 모듈이 nn.Linear인 경우
            elif isinstance(m, nn.Linear):
                '''
                # 작은 숫자로 초기화하는 방법
                # 가중치를 평균 0, 편차 0.02로 초기화합니다.
                # bias를 0으로 초기화합니다.
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
                
                # Xavier Initialization
                # 모듈의 가중치를 xavier normal로 초기화합니다.
                # bias를 0으로 초기화합니다.
                #nn.init.xavier_normal_(m.weight.data)
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0) 
                '''
                # Kaming Initialization
                # 모듈의 가중치를 kaming he normal로 초기화합니다.
                # bias를 0으로 초기화합니다.
                #nn.init.kaiming_normal_(m.weight.data)
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            
            #모듈이 nn.BatchNorm2d or nn.BatchNorm1d 인 경우
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)    
    
    # Convolution --> Batch Normalization --> Activation
    # --> Dropout --> Pooling 순서로 네트워크를 구성하는 것이 일반적임
    def forward(self, x):
        
        #Building Block 1
        x = self.conv1(x) # in_img_size=(28,28), in_channels=1, 
                          # out_channels=256, kernel_size=3, padding=1, out_img_size=(28,28)
        x = self.bn2d1(x) #2D batch normalization layter
        x = self.relu(x)  # in_img_size=(28,28), out_channels=256, out_img_size=(28,28)
        #x = self.dropout2d(x)
        x = self.pool(x) # in_img_size=(28,28), in_channels=256, kernel_size=2, stride=2
                          # out_channels=256,out_img_size=(14,14)
        
        #Building Block 2 
        x = self.conv2(x) # in_img=(14,14), in_channels=256, out_channels=128, kernel_size=3, stride=1
                          # out_img_size=(14,14), out_channels=128
        x = self.bn2d2(x) #2D batch normalization layter
        x = self.relu(x) # out_img_size=(14,14), out_channels=128
        #x = self.dropout2d(x)
        x = self.pool(x) # in_img_size=(14,14), out_channels=128, kernel_size=2, stride=2
                          # out_img_size=(7,7), out_channels=64
        
        #Building Block 3 
        x = self.conv3(x) # in_img=(7,7), in_channels=128, out_channels=64, kernel_size=3, stride=1
                          # out_img_size=(7,7), out_channels=64
        x = self.bn2d3(x) #2D batch normalization layter
        x = self.relu(x) # out_img_size=(7,7), out_channels=64
                                             
        #Serialization for 2D image * channels                           
        x = self.flatten(x) # in_img_size=(7,7), in_channels=64
                            # out_img_size=(3136,)
                            
        #Fully connected layers
        x = self.linear1(x) #in_features=3136, out_features=256
        x = self.relu(x) #in_features=256, out_features=256
        #x = self.dropout1d(x)
        x = self.bn1d1(x) #1D batch normalization layter
        
        #output layer
        x = self.linear2(x) #in_features=256, out_features=10
        x = self.softmax(x) #in_features=10, out_features=10
        return x


#5. optimizer 및  모델 생성

model_dict = { } # optimizer 별 모델 관리 딕션어리 생성
loss_dict = { } # optimizer 별 모델의 loss 값 관리 딕션어리 생성
accuracy_dict = { } # optimizer 별 모델의 accuracy 값 관리 딕션어리 생성

optimizer_case = ['SGD','Adam','AdaGrad','RMSprop'] # 4종류 optimizer
for key in optimizer_case:
    model_dict[key] = CnnNetwork().to(device) # 동일 CNN모델로 설정
    loss_dict[key] = [] # optimizer 별 epoch의 loss 저장하기 위한 위한 리스트 생성 
    accuracy_dict[key] = [] # optimizer 별 epoch의 accuracy 저장하기 위한 리스트 생성 


#4가지 optimizer 정의

optimizer_dict = {}
optimizer_dict['SGD'] = optim.SGD(model_dict['SGD'].parameters(),lr = learning_rate, weight_decay=weight_decay)
optimizer_dict['Adam'] = optim.Adam(model_dict['Adam'].parameters(),lr= 0.001, weight_decay=weight_decay)
optimizer_dict['AdaGrad'] = optim.Adagrad(model_dict['AdaGrad'].parameters(), lr=0.001, weight_decay=weight_decay)
optimizer_dict['RMSprop'] = optim.RMSprop(model_dict['RMSprop'].parameters(),lr=0.001, weight_decay=weight_decay)

#모델을 학습하려면 손실 함수(loss function) 와 옵티마이저(optimizer)가 필요합니다.

loss_fn = nn.CrossEntropyLoss()

#각 학습 단계(training loop)에서 모델은 (배치(batch)로 제공되는) 학습 데이터셋에 
#대한 예측을 수행하고, 예측 오류를 역전파하여 모델의 매개변수를 조정합니다.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_cost = 0
    iterator = tqdm.tqdm(dataloader) # ➊ 학습 로그 출력
    for X, y in iterator:
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_cost += loss
        iterator.set_description(f"loss:{loss.item():.6f}")            
    return train_cost/num_batches     

#모델이 학습하고 있는지를 확인하기 위해 테스트 데이터셋으로 모델의 성능을 확인합니다.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # inference 모드로 실행하기 위해 학습시에 필요한 Drouout, batchnorm등의 기능을 비활성화함
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): # autograd engine(gradinet를 계산해주는 context)을 비활성화함
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct

#학습 단계는 여러번의 반복 단계 (에폭(epochs)) 를 거쳐서 수행됩니다. 각 에폭에서는 
#모델은 더 나은 예측을 하기 위해 매개변수를 학습합니다. 각 에폭마다 모델의 정확도(accuracy)와 
# 손실(loss)을 출력합니다. 에폭마다 정확도가 증가하고 손실이 감소하는 것을 보려고 합니다.

for opt_name, optimizer in optimizer_dict.items():
    print(f'\nOptimizer Name:{opt_name}')
    print(f"--------------------------------\n")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        cost=train(train_dataloader, model_dict[opt_name], loss_fn, optimizer)
        _,correct = test(test_dataloader, model_dict[opt_name], loss_fn)

        loss_dict[opt_name].append(cost.to('cpu').detach().numpy())
        accuracy_dict[opt_name].append(correct*100)
        print(f"epoch : {t+1} | loss : {cost:.6f}")
        print(f"Accuracy : {correct*100:.2f}")

print("Training Done!")



#6.  cost and accuracy graph 그리기

markers = {'SGD' : 'o', 'Adam' : 'x','AdaGrad' : 's', 'RMSprop' : 'D' }
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)

epos = np.arange(1,epochs+1)
for key in optimizer_case:
    plt.plot(epos,loss_dict[key], marker = markers[key], markevery=100, label = key)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.subplot(1,2,2)
for key in optimizer_case:
    plt.plot(epos, accuracy_dict[key],marker = markers[key], markevery=100, label=key)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()