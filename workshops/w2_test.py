#  1. สร้างพจนานุกรมคำพ้องความหมาย และ Data Augmentation
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

LEGAL_SYNONYMS = {
"ละเมิด":["ฝ่าฝืน","กระทำผิด","ล่วงสิทธิ"],
"จำหน่าย":["ขาย","เผยแพร่","กระจายสินค้า"],
"ปลอมแปลง":["ทำเทียม","เลียนแบบ"]
}

def augment_legal_text(text):
    words = text.split()
    new_words = words.copy()
    for i , word in enumerate(words):
        if word in LEGAL_SYNONYMS:
            new_words[i] = random.choice(LEGAL_SYNONYMS[word])
    return " ".join(new_words)
original = "จำเลย ละเมิด และ จำหน่าย สินค้า"
augmented = augment_legal_text(original)
# print(f"---Data Augmentation---")
# print(f"Original : {original}")
# print(f"Augmented : {augmented}")


# 2. SMOTE with Fallback
import numpy as np
from imblearn.over_sampling import SMOTE ,RandomOverSampler
from collections import Counter
def balance_legal_data(X,y):
   counts = Counter(y)
#    print(f"Original distribution : {counts}")
   # คลาสน้อยสุดมีกี่ตัว
   min_samples = min(counts.values())
   if min_samples > 1 :
       sampler = SMOTE(k_neighbors=min(5, min_samples-1),random_state=2)
   else:
       sampler = RandomOverSampler(random_state=2)
   X_res , y_res = sampler.fit_resample(X,y)
#    print(f"Balanced distribution: {Counter(y_res)}")
   return X_res , y_res

# จำลองข้อมูล Imbalance (class 0 = 10 , class 1 = 2)
X_mock = np.random.randn(12,5)
y_mock = np.array([0]*10 + [1]*2)
X_res, y_res = balance_legal_data(X_mock,y_mock)
       
# 3.1 BiLSTM 
import torch 
import torch.nn as nn

class LegalBiLSTM(nn.Module):
    def __init__(self,input_dim=5, hidden_dim=32,output_dim=3): # Changed input_dim to 5
        super(LegalBiLSTM,self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2,output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, x):
        # Removed x.unsqueeze(1) as input should already be (batch_size, seq_len=1, input_dim)
        lstm_out, _ = self.lstm(x)
        # Mean Pooling
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)
model = LegalBiLSTM()
sample_input = torch.randn(1,5,16) # 1 doc 5 token 16 dim (This sample_input is now inconsistent with the model's new input_dim=5 but it's just a print example, not used for training)
output = model(sample_input[:,0:1,:5]) # Adjusted sample input to match new model input_dim
print(f"--BiLSTM Output--")
print(f"Logics: {output.detach().numpy()}")

# 3.2 LSTM with Attention
class LegalLSTMAttention(nn.Module):
    def __init__(self,input_dim=5, hidden_dim=32, output_dim=3): # Changed input_dim to 5
        super(LegalLSTMAttention,self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True,bidirectional=False)
        self.attn_fc = nn.Linear(hidden_dim,1)
        self.fc = nn.Linear(hidden_dim,output_dim)
        nn.init.xavier_uniform_(self.attn_fc.weight)
        nn.init.xavier_uniform_(self.fc.weight) #เลือกค่า weights แบบ Xavier ในการเทรนโมเดลรอบแรก
    def forward(self,x):
        # Removed x.unsqueeze(1) as input should already be (batch_size, seq_len=1, input_dim)
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)
#3.3 เทรนโมเดล LSTM VS BiLSTM
def train_and_evaluate(model, X_train_np, y_train_tensor, X_test_np, y_test_tensor, epochs=100):
    print(f"---Training {model.__class__.__name__}---")

    # Convert numpy arrays to torch tensors and ensure correct dtype
    # Add a sequence length dimension (1) as each input is a single feature vector
    X_train = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(1)
    y_train = y_train_tensor
    y_test = y_test_tensor

    #cost-sensitive weight(FN= false negative)
    # 0 ไม่ผิด 1 ละเมิดสิทธบัตร 2 ละเมิดลิขสิทธิ์
    weights = torch.tensor([1.0, 5.0, 2.0]) # ปรับค่า weights ตามความสำคัญของแต่ละคลาส    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #sample training loop
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    #evaluate with cm
    model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(model(X_test), dim=1) # Removed .numel()
    cm = confusion_matrix(y_test.cpu(), y_pred.cpu())
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['no-ing','patent','copyright'],
                yticklabels=['no-ing','patent','copyright'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

#confusion matrix visualization
# รันเปรียบเทียบโมเดล training & evaluation
class_list = ['no-ing','patent','copyright']

# x_train, y_train
print(f"X_res shape: {X_res.shape}, y_res shape: {y_res.shape}  Balanced distribution: {Counter(y_res)} ")
report_lstm = train_and_evaluate(LegalLSTMAttention(), X_res, torch.tensor(y_res), X_res, torch.tensor(y_res))
report_bilstm = train_and_evaluate(LegalBiLSTM(), X_res, torch.tensor(y_res), X_res, torch.tensor(y_res))