from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from train import X_test, y_test
from data_preprocessing import y_scale
model_name = 'stock_price_GRU'
model = load_model("{}.h5".format(model_name))
print("MODEL-LOADED")
score = model.evaluate(X_test, y_test)
print('Score: {}'.format(score))
yhat = model.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
y_test = y_scale.inverse_transform(y_test)
plt.plot(yhat[-100:], label='Predicted')
plt.plot(y_test[-100:], label='Ground Truth')
plt.legend()
plt.show()
