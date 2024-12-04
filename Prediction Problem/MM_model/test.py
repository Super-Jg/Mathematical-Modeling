import numpy as np
import MM_model
import LSTM
import Dense

Model = MM_model.model(epoch=1000,lr=0.01)
Model.compile(input_dim=1)


lstm1 = LSTM.LSTM('lstm1')
Model.add(lstm1,output_dim=1)

dense1 = Dense.Dense('dense1')
Model.add(dense1,output_dim=5)

dense2 = Dense.Dense('dense2')
Model.add(dense2,output_dim=1)


Model.fit(np.arange(5).reshape(-1,1),np.arange(5).reshape(-1,1)*2)

ans = Model.train()

Model.summary()

prediction = Model.predict(np.arange(5).reshape(-1,1))


import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(np.arange(5),np.arange(5).reshape(-1,1)*2)
plt.plot(np.arange(5),prediction)
plt.show()

