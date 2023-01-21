import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

device = 'cpu'
#%%
df = pd.read_csv('c:/users/armin/desktop/water_potability.csv')

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df_imp_mean = imp_mean.fit_transform(df)

imputer = KNNImputer(n_neighbors=2, weights="uniform")
df_imputer = imputer.fit_transform(df)

df = pd.DataFrame(df_imp_mean, columns=[df.columns[i] for i in range(10)])

class water_quality(nn.Module):
    def __init__(self):
        super(water_quality, self).__init__()
        self.ln1 = nn.Linear(9, 8, dtype=torch.float64)
        self.ln2 = nn.Linear(8, 7, dtype=torch.float64)
        self.ln3 = nn.Linear(7, 6, dtype=torch.float64)
        self.ln4 = nn.Linear(6, 5, dtype=torch.float64)
        self.ln5 = nn.Linear(5, 4, dtype=torch.float64)
        self.ln6 = nn.Linear(4, 3, dtype=torch.float64)
        self.ln7 = nn.Linear(3, 2, dtype=torch.float64)
        self.m = nn.Dropout(p=0.2)
        self.s = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(18, affine=False)
        
    def forward(self, x):
        ln1 = self.ln1(x)
        ln2 = self.ln2(ln1)
        ln3 = self.ln3(ln2)
        ln4 = self.ln4(ln3)
        ln4 = self.m(ln4)
        ln5 = self.ln5(ln4)
        ln6 = self.ln6(ln5)
        ln7 = self.ln7(ln6)
        return self.s(ln7)
    

class animal_dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        
    def __getitem__(self, idx):
        _row = (self.df.iloc[idx]).to_numpy()
        return torch.from_numpy(_row[:-1]), torch.tensor(int(_row[-1]), dtype=torch.float64)
    
    def __len__(self):
        return 3276
        

ds = animal_dataset(df)
ds_train, ds_test = torch.utils.data.random_split(ds, [0.9,0.1])

le = len(ds_train)
le2 = len(ds_test)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=True)

net = water_quality().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.001)

net.train()

# print(ds[0])

for i in range(0, 50):
    lossplus = 0
    running_corrects = 0
    for i, (x, y) in enumerate(dl_train):
        optimizer.zero_grad()
        outputs = net.forward(x.to(device))
        _, preds = torch.max(outputs, 1)
        y = y.to(torch.int64)
        loss = criterion(outputs, y.to(device))
        loss.backward()
        optimizer.step()
        lossplus = loss + lossplus
        running_corrects = torch.sum(preds == y.data) + running_corrects
    exp_lr_scheduler.step()
    print(lossplus)
    print(running_corrects / 2949)
#%%
net.eval()
with torch.no_grad():
    lossplus = 0
    running_corrects = 0
    for i, (x, y) in enumerate(dl_test):
        outputs = net.forward(x.to(device))
        _, preds = torch.max(outputs, 1)
        y = y.to(torch.int64)
        loss = criterion(outputs, y.to(device))
        lossplus = loss + lossplus
        running_corrects = torch.sum(preds == y.data) + running_corrects
print(lossplus)
print(running_corrects / 327)    
#%%
len(ds_test)