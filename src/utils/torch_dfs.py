
from torch.utils import data
import torch
import numpy as np 


class LobDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
            
        x = self._prepare_x(data)
        y = self._get_label(data)
        x, y = self._data_classification(x, y, self.T)
        y = y[:,self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1).float()
        self.y = torch.from_numpy(y).short()

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]
    
    @staticmethod
    def _prepare_x(data):
        df1 = data[:40, :].T
        return np.array(df1)
    
    @staticmethod
    def _get_label(data):
        lob = data[-5:, :].T
        return lob
    
    @staticmethod
    def _data_classification(X, Y, T):
        [N, D] = X.shape
        df = np.array(X)

        dY = np.array(Y)

        dataY = dY[T - 1:N]

        dataX = np.zeros((N - T + 1, T, D))
        for i in range(T, N + 1):
            dataX[i - T] = df[i - T:i, :]

        return dataX, dataY