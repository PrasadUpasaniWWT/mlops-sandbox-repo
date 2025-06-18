import os
import joblib
import pandas as pd
from sagemaker.serve.spec.inference_spec import InferenceSpec

class MyCustomSpec(InferenceSpec):
    def load(self, model_dir):
        self.model = joblib.load(os.path.join(model_dir, "model.pkl"))

    def invoke(self, input_data):
        df = pd.DataFrame(input_data)
        preds = self.model.predict(df)
        return {"prediction": preds.tolist()}
