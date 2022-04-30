import pandas as pd
import numpy as np

# takes in data that is relevant to the user data. 
user_order = pd.read_csv('orders.csv')
# this data will be used to train the program for customer data
training_data = pd.read_csv('order_products__train.csv')
# this data takes prior orders to compare
prior_orders = pd.read_csv('order_products__prior.csv')

# we are merging training data with user order and prior orders with user order to create a new dataframe. 
training_data = training_data.merge(user_order, on='order_id', how='left')
prior_orders = prior_orders.merge(user_order, on='order_id', how='left')

# appending training data to prior orders and concatonating them in order to give us new dataframe (new_df)
new_df = training_data.append(prior_orders, ignore_index=True)
training_data = pd.concat([prior_orders])

# creating individual customer data from prior dataframe. will help individualize each set. 
customer_data_set = new_df[new_df['user_id'].isin(user_order[user_order['eval_set']=='test']['user_id'])] \
                           .drop_duplicates(['user_id', 'product_id'])[['user_id', 'product_id']]

# takes relevant data for each customer data set. 
customer_data_set = customer_data_set.merge(user_order[user_order['eval_set']=='test'], on='user_id', how='left')

# creating a "target" in order to sort the data specifically. 
new_df['target'] = 0
# trains target
new_df.loc[new_df['eval_set'] == 'training_data', 'target'] = 1
# tests target
new_df.loc[new_df['eval_set'] == 'test', 'target'] = 2
# removes user product pairs specifically, not relevant to training data. 

new_df = new_df[~((new_df['target']==1) & (new_df['reordered']==0))]


input("Press Enter to continue...")