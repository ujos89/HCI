import pathlib
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
dataset = pd.read_pickle("../data/analysis/201012/data_prepared.pkl")
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#dataset에 label도 포함되어 있으면 분리하기
train_labels = train_dataset.pop('label')
test_labels = test_dataset.pop('label')

#만약 normalization하면 이 부분 쓰게됨
#def norm(x):
#  return (x - train_stats['mean']) / train_stats['std']
#normed_train_data = norm(train_dataset)
#normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'), 
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    #layers.Dense(16, activation='softmax'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
#model 정보 출력
model.summary()
#model의 정상 작동 확인
example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
example_result
#model을 10번 훈련하는 과정. 하나의 epoch마다 '.'(점)하나가 찍힘
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
EPOCHS = 10

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
  
#.save_weights는 weight를 저장함.
model.save_weights('model_EMO2.h5')
#.save는 model자체를 저장함.
model.save('model_EMO2.model')

#model의 훈련 과정을 시각화함.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
#model의 훈련 과정을 그래프화함.
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure(figsize=(8,12))
  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.scatter(hist['epoch'], hist['mae'], label='Train Error')
  plt.scatter(hist['epoch'], hist['val_mae'], label = 'Val Error')
  plt.ylim([0,.5])
  plt.legend()
  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.scatter(hist['epoch'], hist['mse'], label='Train Error')
  plt.scatter(hist['epoch'], hist['val_mse'], label = 'Val Error')
  plt.ylim([0,.5])
  plt.legend()
  plt.show()
plot_history(history)
#model을 이용해서 test data를 돌려봄.
test_predictions = model.predict(test_dataset).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

#예측의 오차분포를 히스토그램으로 표현.
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

#print error
print(test_labels)
print(test_predictions)
