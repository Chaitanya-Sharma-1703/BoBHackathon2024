import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(openai_api_type="azure",azure_deployment="test-autogen",azure_endpoint="https://autogen-integration.openai.azure.com/",api_version="2024-05-13",api_key="6ba40d064cee4a8baf3d7d61035286fa",temperature=0.5)
agent_executer = create_csv_agent(llm, 'Iris.csv', verbose=True,allow_dangerous_code=True)
result=agent_executer.invoke("give me the data frame with species 'iris-sentosa'?")
print(result['output'])