from torch.utils.data import Dataset
import pandas as pd


class CustomerData(Dataset):
    def __init__(self, pickle_filename):
        super(CustomerData, self).__init__()
        # read the dataframe in
        self.data = pd.read_pickle(pickle_filename)
        # shuffle the data
        # self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[[index]]
        label = row['customerId'].values[0]
        x = row.drop(row.columns[0], axis=1).values[0]
        return x, label
