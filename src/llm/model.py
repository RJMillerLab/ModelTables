"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: This script contains querying OpenAI models and ollama models.
"""
import requests, json, os
import openai
import logging
import tenacity as T
from dotenv import load_dotenv

def load_json(filename: str) -> dict:
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def setup_openai(fname, mode='azure'):
    assert mode in {'openai', 'azure'}
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-test')
    if mode == 'openai':
        openai.api_type = "open_ai"
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = OPENAI_API_KEY
        secrets = None
    else:
        #openai.api_version = "2023-03-15-preview"
        secrets = load_json(fname)
        openai.api_type = "azure"
        openai.api_base = secrets['MS_ENDPOINT']
        openai.api_key = secrets['MS_KEY']
    return secrets

@T.retry(stop=T.stop_after_attempt(5), wait=T.wait_fixed(60), after=lambda s: logging.error(repr(s)))
def query_openai(prompt, mode='azure', model='gpt-35-turbo', max_tokens=1200, **kwargs):
    if mode == 'openai':
        response = openai.chat.completions.create(model=model,
                                             messages=[{'role': 'user', 'content': prompt}],
                                             max_tokens=max_tokens,
                                             **kwargs
                                             )
    else:
        response = openai.chat.completions.create(
            deployment_id=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            **kwargs,
        )
    return response.choices[0].message.content

def generate_completion_stream(model_name, prompt):
    # Requires Ollama downloaded!
    api_url = "http://localhost:11434/api/generate"  # Replace with the actual API URL if different
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
    }
    # Making a request with streaming enabled
    response = requests.post(api_url, headers=headers, data=json.dumps(payload), stream=True)
    final_response = ""
    if response.status_code == 200:
        for line in response.iter_lines():
            # Ensure the line has content
            if line:
                decoded_line = json.loads(line.decode('utf-8'))
                # Check if 'response' key exists and append its content
                if 'response' in decoded_line:
                    final_response += decoded_line['response']
                    #print(decoded_line['response'], end='', flush=True)  # Print each character without newline
        #print("\nComplete response:", final_response)
        return final_response
    else:
        print(f"Error: {response.status_code}")
        return response.status_code

def LLM_response(chat_prompt,llm_model="gpt-3.5-turbo-0125",history=[],kwargs={},max_tokens=1000): # "gpt-4-0125-preview"
    """
    get response from LLM
    """
    if llm_model.startswith('gpt-3.5') or llm_model.startswith('gpt-4') or llm_model.startswith('gpt3.5') or llm_model.startswith('gpt4'):
        setup_openai('', mode='openai')
        response = query_openai(chat_prompt, mode="openai", model=llm_model, max_tokens=max_tokens)
        history.append([chat_prompt, response])
    elif llm_model in ['llama3','llama2','mistral','dolphin-phi','phi','neural-chat','starling-lm','codellama','llama2-uncensored','llama2:13b','llama2:70b','orca-mini','vicuna','llava','gemma:2b','gemma:7b']:
        # use ollama instead, required ollama installed and models downloaded, https://github.com/ollama/ollama/tree/main?tab=readme-ov-file
        response = generate_completion_stream(llm_model, chat_prompt)
        history.append([chat_prompt, response])
    else:
        raise NotImplementedError
    return response, history

if __name__=='__main__':
    #llm_model = "dolphin-phi"
    #llm_model = "gpt-3.5-turbo-0125"
    llm_model = "gpt-4-turbo"
    prompt = "Provide the python code for computing the neighborhood graph on data with the API only from Ehrapy. Apply it to the built-in dermatology dataset."
    response, history = LLM_response(prompt, llm_model, max_tokens=1000)
    print(f'User: {prompt}')
    print(f'LLM: {response}')
