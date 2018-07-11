import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import scipy.stats
import scipy.stats
from sklearn.preprocessing import MinMaxScaler


def load_coin_prices(file_name):
	#Date, Open, High, Low, Close, Volume, Market Cap

	df = pd.read_csv(file_name+".csv")

	arr = df.as_matrix()

	data = []
	for i in arr:
		data.append([float(j) for j in i[1:5]])
	return np.array(data)

def calc_price_change(data):
	res = [[] for _ in range(len(data)-1)] 
	for i in range(len(data)-1):
		for j in range(len(data[i])):
			res[i].append(data[i+1][j] - data[i][j])
	return res

def shape_for_lstm(data, look_back):
	product = []
	alive = True
	alive_iter = 1
	temp = []

	for i in data:
	    if alive:
	        temp.append(i) 
	        alive_iter += 1
	    if alive_iter == look_back:
	        alive = False

	    if not alive:
	        product.append(temp)
	        temp.append(i)
	        temp = temp[1:]

	return np.array(product)

def get_labels(data, look_back):
	return np.array([i[look_back-1][1] for i in data[1:]])

def split_data(data, labels, split_point):
	x1 = data[:split_point]
	x2 = data[split_point:]

	y1 = labels[:split_point]
	y2 = labels[split_point:] 

	return x1,x2,y1,y2

def train_model(x, y, look_back, num_layers, num_cells, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(num_cells, activation='hard_sigmoid', input_shape=(look_back, 4)))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print("Training Model")
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)
    print("Trained")
    return model

def run_neural_net(look_back, num_layers, num_cells, epochs, batch_size):
	data = load_coin_prices("dash_price") 
	data = calc_price_change(data) 
	data = shape_for_lstm(data, look_back)

	print(data[:2])

	labels = get_labels(data, look_back)

	data = data[:len(data)-1]
	 
	x1, x2, y1, y2 = split_data(data, labels, int(0.9*len(data))) 

	model = train_model(x1, y1, look_back, num_layers, num_cells, epochs, batch_size)

	predictions = model.predict(x2) 

	arr1 = [i for i in y2]
	arr2 = [i[0] for i in predictions]

	print(scipy.stats.pearsonr(arr1, arr2)) 

	print("Real ---- Predicted")
	display_count = 0
	for (i, j) in zip(arr1, arr2):
		print(i, j)
		if display_count == 10: break
		display_count += 1
	return zip(arr1,arr2)


#HYPER PARAMS 
look_back = 15
epochs = 100
batch_size = 10
num_cells = 100
num_layers = 1

predictions = run_neural_net(look_back, num_layers, num_cells, epochs, batch_size)

f = open("predictions.txt", "w")

for i in predictions:
	f.write(str(i[0])+" "+str(i[1])+"\n")

#[[728.99 738.74 685.04 685.04]
# [687.7  731.11 687.22 728.73]]