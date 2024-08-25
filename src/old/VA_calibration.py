
# load packages
import numpy as np
import torch
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, precision_score, recall_score, f1_score
from custom_calibrators.venn_abers import VennAbersMultiClass

from model.DeepLOB import deeplob
from utils.torch_dfs import LobDataset
from utils.constants import DEVICE


dec_data = np.loadtxt('data/Train_Dst_NoAuction_DecPre_CF_7.txt')
dec_cal = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

batch_size = 64

dataset_cal = LobDataset(data=dec_cal, k=4, num_classes=3, T=100)
cal_loader = torch.utils.data.DataLoader(dataset=dataset_cal, batch_size=batch_size, shuffle=False)

print('Calibration Data Shape:', dataset_cal.x.shape, dataset_cal.y.shape)

del dec_cal, dec_data, dataset_cal

dec_test1 = np.loadtxt('data/Test_Dst_NoAuction_DecPre_CF_7.txt')
dec_test2 = np.loadtxt('data/Test_Dst_NoAuction_DecPre_CF_8.txt')
dec_test3 = np.loadtxt('data/Test_Dst_NoAuction_DecPre_CF_9.txt')
dec_test = np.hstack((dec_test1, dec_test2, dec_test3))



dataset_test = LobDataset(data=dec_test, k=4, num_classes=3, T=100)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)


print('Test Data Shape:',dataset_test.x.shape, dataset_test.y.shape)

del dec_test, dec_test1, dec_test2, dec_test3, dataset_test


model = deeplob(y_len = 3)
model_path = 'model/best_val_model_pytorch'

model = torch.load(model_path,  map_location=torch.device(DEVICE))
model.eval()

# After fitting the calibrator
calibrator = VennAbersMultiClass(model, cal_size=0.2)
calibrator.fit(cal_loader)



# Get all statistics
stats = calibrator.get_statistics(test_loader, 0.1)

# Print the statistics
for stat_name, stat_value in stats.items():
    print(f"{stat_name}: {stat_value:.4f}")
