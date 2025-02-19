import pandas as pd
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

input_file_path = "datasetI.csv"
output_file_path = "transfer_datasetI.csv"

df = pd.read_csv(input_file_path)
df=df[:31]

llm = ChatOpenAI(
    model="qwen-turbo-1101",
    temperature=0.9,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# Define the prompt template
template_string = """Transfer the text \
that is delimited by triple backticks \
into a style that describes isolation instead of reachability. \
text: ```{text}```"""

# Initialize the prompt template
prompt_template = ChatPromptTemplate.from_template(template_string)

# Add a new column for the transfer_text
df['transfer_text'] = ''

for index, row in df.iterrows():
    # Generate the prompt with the current text
    human_response = llm(prompt_template.format_messages(text=row['human_language']))
    # Save the response content into the DataFrame
    df.at[index, 'transfer_text'] = human_response.content
    print(f"Response for entry {index}: {human_response.content}")

# Save the DataFrame to CSV
df.to_csv(output_file_path, index=False)
print(f"Translated results have been saved to {output_file_path}")

