import pandas as pd
import numpy as np
Customer = pd.read_csv("F:\All Assignment\MLDA\pandas\Customer.csv")
#print(Customer)
Product_hierarchy = pd.read_csv("F:\All Assignment\MLDA\pandas\prod_cat_info.csv")
#print(Product_hierarchy)
Transaction = pd.read_csv("F:\All Assignment\MLDA\pandas\Transactions.csv")
#print(Transaction)
Customer_Trans = pd.merge(left = Customer,
                          right = Transaction,
                          left_on = 'customer_Id',
                          right_on = 'cust_id',
                          how = 'inner',
                          indicator = True)
#print(Customer_Trans)
Customer_Final = pd.merge(left = Customer_Trans,
                          right = Product_hierarchy,
                          left_on = 'prod_cat_code',
                          right_on = 'prod_cat_code',
                          how = 'inner'
                          )
#print(Customer_Final)
#print(Customer_Final.dtypes)
'''
Data_min = Customer_Final['total_amt'].min()
Data_max = Customer_Final['total_amt'].max()
Data_q1  = np.percentile(Customer_Final.total_amt,25)
median  = np.percentile(Customer_Final.total_amt,50)
Data_q3  = np.percentile(Customer_Final.total_amt,75)
print('Min = ',Data_min)
print('Max = ',Data_max)
print('Median = ',median)
print('Q1 = ',Data_q1)
print('Q3 = ',Data_q3)
'''
'''
freq_table = pd.crosstab(index = Customer_Final['Gender'],
                         columns = Customer_Final['Store_type'])
freq_table.columns = ['TeleShop','MBR','e-shop','Flagshipstore']
freq_table.index = ['Male','Female']
print(freq_table)
'''
'''
freq_table = pd.crosstab(index = Customer_Final['Gender'],
                         columns = Customer_Final['prod_cat'])
freq_table.columns = ['Books','Bags','Clothing','Footwear','Electronics','Home and kitchen']
freq_table.index = ['Male','Female']
print(freq_table)

freq_table = pd.crosstab(index = Customer_Final['Gender'],
                         columns = Customer_Final['prod_subcat'])
freq_table.columns = ['Men','Women','Kid','Mobile','Computer','Personal Appliances','Cameras','Audio and video',
                      'Fiction','Academic','Non-fiction','Children','Comics','DIY','Furnishing','Kitchen',
                      'Bath','Tools']
freq_table.index = ['Male','Female']
print(freq_table)
'''
import matplotlib.pyplot as plt
'''

Tax = Customer_Final['Tax']
plt.hist(Tax,color=['yellow'])
plt.xlabel('tax')
plt.ylabel('Frequency')
plt.show()

Total_Amt = Customer_Final['total_amt']
plt.hist(Total_Amt,color = 'Blue')
plt.xlabel('Total amount')
plt.ylabel('Frequency')
plt.show()

print('COUNT')
mf_count=Customer_Final['Gender'].value_counts()
print(mf_count.values)
plt.bar(mf_count.index,mf_count.values)
plt.show()
'''

'''
#best 
Customer_Final['Gender'].value_counts().plot('bar')
Customer_Final['Store_type'].value_counts().plot('bar')
Customer_Final['prod_cat'].value_counts().plot('bar')
Customer_Final['prod_subcat'].value_counts().plot('bar')
'''
'''
df = Customer_Final['total_amt']
count2 = Customer_Final.loc[(df<0),['total_amt']].count()
print(count2)
'''

'''
# Popular among Male
M = Customer_Final.loc[Customer_Final['Gender']=='M']
group_prod = M.groupby(['prod_cat'])['total_amt'].sum()
popular_M = group_prod.nlargest(1)
print('The most popular product category in Male customers is : ',popular_M)
'''
'''
max_cust = Customer['city_code'].value_counts()
t = max_cust.nlargest(1)
print("City code which has Maximum customers is : ",t)
'''
#percentage of customers from city code 3
tot_customer = Customer['customer_Id'].count()
percent = round((595/tot_customer)*100,2)
print("Percentage of customers from the city code 3 is {}% : ".format(percent))
'''
sort_list = Customer_Final.sort_values(['total_amt','Qty'],ascending = False)
print(sort_list.head(1)['Store_type'])

df = pd.DataFrame(Customer_Final)
tf = df[df.prod_cat.isin(['Electronics','Clothing']) & (df.Store_type == 'Flagship store')]
total = tf.total_amt.sum()
print('Total amount earned',total)


df = pd.DataFrame(Customer_Final)
tf1 = df[(df.Gender == 'M') & (df.prod_cat == 'Electronics')]
total = tf1.total_amt.sum()
print('Total amount earned',total)

df = pd.DataFrame(Customer_Final)
tf = df[df.prod_cat.isin(['Electronics','Clothing']) & (df.Store_type == 'Flagship store')]
total = tf.total_amt.sum()
print('Total amount earned',total)


df = pd.DataFrame(Customer_Final)
tf1 = df[(df.Gender == 'M') & (df.prod_cat == 'Electronics')]
total = tf1.total_amt.sum()
print('Total amount earned',total)

df = pd.DataFrame(Customer_Final)
df1 = df[(df.total_amt > 0)]
ts = df1.transaction_id.nunique()
print('Total customers having more than 10 unique transactions are - ',ts)


df = pd.DataFrame(Customer_Final)
curr_year = pd.to_datetime('today').year
dob_year = pd.DatetimeIndex(df['DOB']).year          #extract year from DOB
x = dob_year-100                                               # for the years which belongs to 60's
v = curr_year - x
y = curr_year - dob_year
 
df['age'] = (np.where(dob_year > curr_year,v,y))
print(df)
'''
 
