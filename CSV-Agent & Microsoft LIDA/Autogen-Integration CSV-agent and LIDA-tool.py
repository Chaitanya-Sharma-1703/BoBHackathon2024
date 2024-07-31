import autogen
from typing import Annotated,Literal
import os
import pandas as pd
import seaborn as sns
from langchain.agents import AgentExecutor
import openai
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from dotenv import load_dotenv
from lida import Manager, TextGenerationConfig, llm
from lida.utils import plot_raster
from llmx.generators.text.openai_textgen import OpenAITextGenerator
from llmx.generators.text.textgen import sanitize_provider
from langchain_experimental.agents import create_csv_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()
dataset=pd.read_csv('./Iris.csv')
config_list=[
    {
        'model': 'test-gpt-4o',
        'api_key': '525928007ca5489eb783a2e03960cba0',
        'base_url': 'https://test-autogen.openai.azure.com/',
        'api_type': 'azure',
        'api_version': '2023-03-15-preview',
    }
]
llm_config={
    "config_list":config_list,
}
# ,deployment_name='llm_wrapper',model='gpt-3.5-turbo'
def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

def to_summarize():
    
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
        img = base64_to_image(img_base64_string)
        plt.imshow(img)
        #display(img)
    
    return "summary of the dataset"

def ask_csv(query: str):
    llm = AzureChatOpenAI(openai_api_type="azure",azure_deployment="test-gpt-4o",azure_endpoint="https://test-autogen.openai.azure.com/",api_version="2023-03-15-preview",api_key="525928007ca5489eb783a2e03960cba0",temperature=0.5)
    agent_executer = create_csv_agent(llm, 'Iris.csv', verbose=True,allow_dangerous_code=True)
    result=agent_executer.invoke(f"{query}")
    return result['output']

assistant_summarizer=autogen.ConversableAgent(
    name="Data_summarizer",
    system_message="""You are a data summarizer,
    which provide different visualizing goals for a given dataset
    for summarizing the data or generating visualizing goals use the functions you have been provided with""",
    llm_config=llm_config
)
chatbot = autogen.ConversableAgent(
    name="chatbot",
    system_message="""you have a csv file with you,
    this is not a summarizer bot, it can only provide you to communicate with the csv file
    you just have to call a function or tool given to you,
    The function ask_csv takes natural language as the parameter dont write query by yourself, just pass the user query 
    as it is to the function 
    for reading csv files only use the functions you have been provided with.
    Reply TERMINATE when the task is done.""",
    llm_config=llm_config,
)
user=autogen.UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=2,
    code_execution_config={
        "use_docker":False
    },
    function_map={"ask_csv": ask_csv,"to_summarize":to_summarize}
)
group_chat=autogen.GroupChat(
    agents=[assistant_summarizer,chatbot,user],
    messages=[],
    max_round=10,
)
group_chat_manager=autogen.GroupChatManager(
    groupchat=group_chat,
    human_input_mode="ALWAYS",
    llm_config=llm_config
)
assistant_summarizer.register_for_llm(name="to_summarize", description="tool for getting summary of a preloaded dataset")(to_summarize)
chatbot.register_for_llm(name="ask_csv",description="tool for querying csv file")(ask_csv)
user.register_for_execution(name="ask_csv")(ask_csv)
user.register_for_execution(name="to_summarize")(to_summarize)
chat_result=user.initiate_chat(
    group_chat_manager,
    message="""give_me_visualizing_goal_for_the_dataset_?""",
    summary_method="last_msg"
)