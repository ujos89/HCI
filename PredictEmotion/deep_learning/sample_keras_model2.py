import pathlib
import pandas as pd 
import seaborn as sns
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

dataset = pd.read_pickle("./data/analysis/201012/data_prepared.pkl")

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#dataset에 label도 포함되어 있으면 분리하기
train_labels = train_dataset.pop('label')
test_labels = test_dataset.pop('label')

def build_model():
  model = keras.Sequential([
  #actiavtion은 relu, sigmoid, softmax 등을 고려해 볼 수 있음.
  #Dropout(0.5) 을 추가하여 overfitting을 방지할 수도 있음.
  #validation_data의 사용=> test data를 training때 이용하여 model을 evaluate할 수도 있음.
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    #layers.Dense(16, activation='softmax'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  """Alternative compiling
  model.compile(
  	optmizier = 'adam',
  	loss = 'categorical_crossentropy',
  	metrics=['accuracy'])
  	"""
  return model
  
  
model_EMO = build_model()

history = model_EMO.fit(train_dataset, train_labels,
  epochs=10, validation_split = 0.2, verbose=0)
  
  
#.save_weights는 weight를 저장함.
model_EMO.save_weights('model_EMO.h5')
#.save는 model자체를 저장함.
model_EMO.save('model_EMO2.model')
  
#test_predictions = model_EMO.predict(test_dataset).flatten()


