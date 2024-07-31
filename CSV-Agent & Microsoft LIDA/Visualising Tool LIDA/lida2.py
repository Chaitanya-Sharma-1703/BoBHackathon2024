import openai
import os
import pandas as pd
import seaborn as sns
import sys
# import torch
import warnings
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from dotenv import load_dotenv
from lida import Manager, TextGenerationConfig, llm
from lida.utils import plot_raster
from llmx.generators.text.openai_textgen import OpenAITextGenerator
from llmx.generators.text.textgen import sanitize_provider
from PIL import Image
import numpy as np
import io
import cv2

# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    cv2.imshow("Image",opencv_img)
    return opencv_img

dataset=pd.read_csv('./Iris.csv')
def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

def to_summarize(dataset):
    
    text_gen = llm(
    provider='openai',
    api_type="azure",
    azure_endpoint="https://test-autogen.openai.azure.com/",
    api_key="525928007ca5489eb783a2e03960cba0",
    api_version="2023-03-15-preview",
    )
    lida = Manager(text_gen=text_gen)

    textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="test-gpt-4o", use_cache=True)

    summary = lida.summarize(
    dataset,
    summary_method="default", 
    textgen_config=textgen_config
    ) 
    goals = lida.goals(summary, n=2, textgen_config=textgen_config)

    for g in goals:
        #display(g)
        print(g)
        textgen_config = TextGenerationConfig(n=1,model='test-gpt-4o', temperature=0.5, use_cache=True)
        charts = lida.visualize(summary=summary, goal = g, textgen_config=textgen_config, library='seaborn')  
        #charts[0]
        img_base64_string = charts[0].raster
        img = stringToRGB(img_base64_string)
        #display(img)
    

def user_query_viz(user_query):
    text_gen = llm(
    provider='openai',
    api_type="azure",
    azure_endpoint="https://test-autogen.openai.azure.com/",
    api_key="525928007ca5489eb783a2e03960cba0",
    api_version="2023-03-15-preview",
    )
    lida = Manager(text_gen=text_gen)

    textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="test-gpt-4o", use_cache=True)

    summary = lida.summarize(
    dataset,
    summary_method="default", 
    textgen_config=textgen_config
    )
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="test-gpt-4o", use_cache=True)

    charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
    img_base64_string = charts[0].raster
    img = base64_to_image(img_base64_string)
    #display(img)
    plt.imshow(img)
    return img

# user_query_viz("Scatter plot of PetalLengthCm vs PetalWidthCm colored by Species")
to_summarize(dataset)
