# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy a Hugging Face `transformers` model with Model Serving
# MAGIC
# MAGIC This notebook demonstrates how to deploy a model logged using the Hugging Face `transformers` MLflow flavor to a serving endpoint. Using this example you can pull a model from hugging face and set it to a .

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install and import libraries 

# COMMAND ----------

!pip install --upgrade mlflow
!pip install --upgrade transformers
!pip install --upgrade accelerate
%pip install --upgrade "mlflow-skinny[databricks]"

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import requests
import json
from transformers import pipeline
import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize and configure your model
# MAGIC
# MAGIC Define and configure your model using any popular ML framework.

# COMMAND ----------




# import torch
# # Ensure dbutils is available and you have a widget named 'model' set up
model_name = dbutils.widgets.get("model_name")
device = 0 if torch.cuda.is_available() else -1

# Initialize the pipeline with the model specified in the widget, assuming it's a valid model name
text_generation_pipeline = pipeline(task='text-generation', model=model_name, pad_token_id=50256, device=device)  # Use device=0 for GPU or device=-1 for CPU


# COMMAND ----------

# MAGIC %md
# MAGIC ## Log your model using MLflow
# MAGIC
# MAGIC The following code defines inference parameters to pass to the model at the time of inference and defines the schema for the model, before logging the model with the MLflow Hugging Face `transformers` flavor.

# COMMAND ----------

max_new_tokens = dbutils.widgets.get("max_tokens")

temperature = dbutils.widgets.get("temperature")

registered_model_name = dbutils.widgets.get("registered_name")

inference_config = {"max_new_tokens": max_new_tokens, "temperature": temperature}


input_example = pd.DataFrame(["Hello, I'm a language model,"])
output = generate_signature_output(text_generation_pipeline, input_example)
signature = infer_signature(input_example, output, params=inference_config)


with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model = text_generation_pipeline,
        artifact_path = "my_sentence_generator",
        inference_config = inference_config,
        input_example = input_example,
        signature = signature,
        registered_model_name = registered_model_name,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test your model in a notebook
# MAGIC
# MAGIC In the following command, you load the model, so you can generate a prediction with the given parameters.

# COMMAND ----------

# Load the model
my_sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)


my_sentence_generator.predict(
    pd.DataFrame(["Hello, I'm a language model,"]),
    params={"max_new_tokens": 20, "temperature": 1},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure and create your model serving endpoint
# MAGIC
# MAGIC The following variables set the values for configuring the model serving endpoint, such as the endpoint name, compute type, and which model to serve with the endpoint. After you call the create endpoint API, the logged model is deployed to the endpoint.

# COMMAND ----------

# Set the name of the MLflow endpoint

endpoint_name = registered_model_name

# Name of the registered MLflow model
model_name = registered_model_name

# Get the latest version of the MLflow model
model_version = MlflowClient().get_registered_model(model_name).latest_versions[0].version 

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "GPU_MEDIUM" 

# Specify the scale-out size of compute (Small, Medium, Large, etc.)
workload_size = "Small" 

# Specify Scale to Zero(only supported for CPU endpoints)
scale_to_zero = False 

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# send the POST request to create the serving endpoint

data = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": model_version,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
                "workload_type": workload_type,
            }
        ]
    },
}

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(
    url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers
)

print(json.dumps(response.json(), indent=4))
