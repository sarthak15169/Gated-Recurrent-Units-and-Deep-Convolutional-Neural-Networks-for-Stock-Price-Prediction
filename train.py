from sklearn.model_selection import train_test_split
import numpy as np
from data_preprocessing import X, Y
from model import model
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
X_train = X_train.reshape((-1,1,4))
X_test = X_test.reshape((-1,1,4))
#model.fit(X_train,y_train,batch_size=250, epochs=500, validation_split=0.1, verbose=1)
#model.save("{}.h5".format(model_name))
#print('MODEL-SAVED')
