import tensorflow.keras.preprocessing.image as keras_image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import base64
from PIL import Image
import io
import json

class ResnetModel():

    def __init__(self):
        self.loaded = False

    def load(self):
        self.model = load_model('model/resnet50_model.keras')
    
    def base64_to_image(self,base64_encoded_image):
        img_array = base64.b64decode(base64_encoded_image)
        img = Image.open(io.BytesIO(img_array))
        img = img.resize([224,224])
        x = keras_image.img_to_array(img)
        x = preprocess_input(x)
        return x
    
    def predict(self,X,feature_names=None):
        if not self.loaded:
            self.load()
        input = np.array([self.base64_to_image(im) for im in X])
        preds = self.model.predict(input)
        predictions = decode_predictions(preds, top=3)
        
        final_op = []
        for prediction in predictions:
            im_pred = []
            for p in prediction:
                im_pred.append({"Label":p[1],"score":p[2].item()})
            final_op.append(im_pred)
        return json.dumps(final_op)

# if __name__=="__main__":
#     import time  
#     model = ResnetModel()
#     print(type(model.model))
#     image = Image.open('CNN-Resnet-version2/puppy.jpeg')
#     buffered = io.BytesIO()
#     image.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
#     start = time.time()
#     preds = model.predict([img_str])
#     end = time.time()
#     print(end - start)
#     print('Predicted:', preds)


