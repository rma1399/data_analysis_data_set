import psycopg2
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    
conn = psycopg2.connect(
host="localhost",  
database="testing",
user="postgres",
password="Rack10020315#"
)

query = "SELECT * FROM sales"
data = pd.read_sql_query(query, conn)

conn.close()

result = data.groupby('product_name').agg({'price': 'mean', 'quantity': 'sum'}).reset_index()

result.columns = ['product_name', 'avg_price', 'total_quantity']

x = result.drop(columns='product_name')
y = result['product_name']

label = LabelEncoder()
y = label.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32,input_dim=x_train.shape[1], activation='sigmoid'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=200)








