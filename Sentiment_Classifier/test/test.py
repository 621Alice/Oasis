from keras.models import load_model
from functions import *


#Get weights from the model and make json file

model_test = load_model(p+'/model/checkpoint-toxic-texts-0.058.h5')
model_test.save_weights(p+'/model/checkpoint-toxic-texts-0.058-weight-only.h5')
json_model = model_test.to_json()
with open(p+'/model/checkpoint-toxic-texts-0.058.json', 'w') as f:
    f.write(json_model)



# model_test = load_model(p+'/model/checkpoint-5labels-0.928.h5')
# model_test.save_weights(p+'/model/checkpoint-5labels-0.928-weight-only.h5')
# json_model = model_test.to_json()
# with open(p+'/model/checkpoint-5labels-0.928.json', 'w') as f:
#     f.write(json_model)


#load weights and json file to predict
# from keras.models import model_from_json
# json_file = open(p+"/model/checkpoint-5labels-0.928.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights(p+"/model/checkpoint-5labels-0.928-weight-only.h5")
