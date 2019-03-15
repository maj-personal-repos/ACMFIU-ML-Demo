import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from matplotlib import cm
pd.set_option('display.max_columns', 500)


filename = 'data/XAXD_events_all_time.anon.csv'
df = pd.read_csv(filename, dtype='str')

print(df[['customerId','st', 'ClientIPAddress', 'EventTime']].head())

print(df.columns.values)

# read in some pre-prepped data, else create it
try:
    unique_customerIds = pd.read_pickle('unique_customerIds.pickle')
    unique_users_per_customerId_df = pd.read_pickle('unique_users_per_customerId_df.pickle')
except FileNotFoundError:
    print("Data not found... creating.")
    unique_customerIds = df.groupby(df.customerId).count()
    unique_customerIds.to_pickle('unique_customerIds.pickle')
    unique_users_per_customerId = df.groupby(df.customerId).BrokeringUserName.nunique()
    unique_users_per_customerId_df = pd.DataFrame(
        {'Customer Id': unique_users_per_customerId.index, 'Number of Users': unique_users_per_customerId.values})
    unique_users_per_customerId_df.to_pickle('unique_users_per_customerId_df.pickle')

print("Number of unique customer Ids: {}".format(len(unique_customerIds)))
user_slice = unique_users_per_customerId_df.sort_values('Number of Users', ascending=False).head(10)
print(user_slice.head())
user_slice.set_index("Customer Id", drop=True, inplace=True)
user_slice.plot.bar()

records_per_customer_id = df.groupby(df.customerId).count()
print(records_per_customer_id['st'].to_frame().sort_values('st', ascending=False).head(10))

print(records_per_customer_id['st'].to_frame().sort_values('st', ascending=False).tail(10))

customer_list = ['afd5548a3239bf', 'f0a222c53d78c', '952affeb7c5aa61', '8ff7407f547c29']
df_filtered = df[df.customerId.isin(customer_list)]
df_filtered = df_filtered.drop(['prod',
                                'prodVer',
                                'type',
                                'IsNonBrokered',
                                'IsSecureIca',
                                'Protocol',
                                'VdaPowerOnTime',
                                'VdaPoweredOnTime',
                                'VdaRegisteredTime'], axis=1)
df_filtered.head()

filtered_grouped = df_filtered.groupby('customerId')
group_sizes = filtered_grouped.size()
min_group_size = group_sizes.min()
print("The group sizes are \n{}".format(group_sizes))
print("The minimum group size is :{}".format(min_group_size))

try:
    balanced_dataset = pd.read_csv('balanced_dataset.csv', dtype=str)

except FileNotFoundError:

    grouped_sampled = []

    for customerId in customer_list:
        group = filtered_grouped.get_group(customerId)
        grouped_sampled.append(group.sample(frac=min_group_size / group_sizes[customerId]))

    balanced_dataset = pd.concat(grouped_sampled)

    balanced_dataset.to_csv('balanced_dataset.csv')

# let's check to make sure the dataset is balanced

print("The balanced group sizes are: \n{}".format(balanced_dataset.groupby('customerId').size()))
print("The final dataset size is: \n{}".format(len(balanced_dataset.index)))


df = pd.read_csv('balanced_dataset.csv', dtype=str)

df = df[['customerId', 'st']].copy()

df_sm = df.sample(frac=0.2)

df_sm.to_csv('balanced_dataset_sm.csv')

df_sm = pd.read_csv('balanced_dataset_sm.csv', dtype=str)

df = df_sm[['customerId', 'st']].copy()

customer_map = {'afd5548a3239bf': 0, 'f0a222c53d78c': 1, '952affeb7c5aa61': 2, '8ff7407f547c29': 3}

df.customerId = df.customerId.map(customer_map)


def strip_time(date_string):
    if not isinstance(date_string, str):
        print(date_string)
    try:
        timestamp = datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        timestamp = datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')

    return timestamp


def seconds_from_day_start(date_time):
    year = date_time.year
    month = date_time.month
    day = date_time.day
    return (date_time - datetime.datetime(year, month, day)).total_seconds()


df = df.dropna(how='any')

df['st_weekday'] = df.st.apply(lambda x: strip_time(x).date().weekday())

df['st_seconds'] = df.st.apply(lambda x: seconds_from_day_start(strip_time(x)))

scaler = preprocessing.MinMaxScaler()

df['st_weekday'] = pd.DataFrame(scaler.fit_transform(df[['st_weekday']].values.astype(float)), dtype=float)
df['st_seconds'] = pd.DataFrame(scaler.fit_transform(df[['st_seconds']].values.astype(float)), dtype=float)

df.drop(['st'], axis=1, inplace=True)

cmap = cm.get_cmap('Spectral')
df.sample(frac=0.1).plot.scatter(x='st_weekday',
                                 y='st_seconds',
                                 c='customerId',
                                 cmap=cmap,
                                 edgecolor=None,
                                 alpha=0.5,
                                 # s=100,
                                 marker="_")
plt.show()

print(df.head())

train_mask = np.random.rand(len(df)) < 0.7

train_data = df[train_mask]

test_val = df[~train_mask]

test_mask = np.random.rand(len(test_val)) < 2.0 / 3.0

test_data = test_val[test_mask]

val_data = test_val[~test_mask]

print(len(test_data))
print(len(train_data))
print(len(val_data))

train_data.to_pickle('data/train_data.pickle')
test_data.to_pickle('data/test_data.pickle')
val_data.to_pickle('data/val_data.pickle')
