#transformers==5.4.0
#outlines==1.2.12
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
from func_timeout import func_set_timeout
import re
import ujson as json
import google.generativeai as genai
import
import anthropic
from pydantic import BaseModel, Field
from typing import List,Literal
import json
import outlines
from outlines import Generator
import google.generativeai as genai
from transformers import pipeline
import pandas as pd
import os
from google.colab import userdata


LabelType = Literal[
    'travel transport and communication',
    'disease and death',
    'money work and finance',
    'religion',
    'law politics and warfare',
    'social life',
    'general'
]

class ParagraphLabels(BaseModel):
    paragraph: str = Field(description="The paragraph being annotated.")
    labels: List[LabelType]

class open_annotator:
    def __init__(self, engine: str = 'mistral',use_demo:bool=True):
        self.input_format = input_format
        self.output_format = output_format
        self.qwen = "Qwen/Qwen2.5-7B-Instruct"
        self.mistral = "mistralai/Mistral-7B-Instruct-v0.3"
        self.use_demo=use_demo
        if engine=="qwen":
          self.tokenizer = AutoTokenizer.from_pretrained(self.qwen)
          self.model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(self.qwen,dtype=torch.bfloat16, device_map="auto"),
    AutoTokenizer.from_pretrained(self.qwen)
)
        elif engine=="mistral":
           self.tokenizer = AutoTokenizer.from_pretrained(self.mistral)
           self.model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(self.mistral,dtype=torch.bfloat16, device_map="auto"),
    AutoTokenizer.from_pretrained(self.mistral))


    def generate_prompt(self, sample, demo=None):
        to_annotate = self.input_format.format(json.dumps(sample['text']))
        if self.use_demo and demo:
            demo_file = {str(i['id']):i for i in demo}
            demo = [demo_file[str(pointer['id'])] for pointer in reversed(demo_index[str(sample['id'])])]
            demo_annotations = "\n".join(f"{{{self.input_format.format(json.dumps(d['text']))},{self.output_format.format(json.dumps(d['label']))}}}" for d in demo)
            return f"Here are some examples to guide the labeling:\n{demo_annotations}\n Now please annotate the following paragraph using topic labels:\n{to_annotate}"
        else:
            return f"please annotate the following paragraph using topic labels and return the output in the following JSON format:{{“paragraph”: paragraph, “labels”: [topics,..]}}:\n{to_annotate}"

    @func_set_timeout(60)
    def online_annotate(self, sample, demo=None): #to annotate
        annotation_prompt = self.generate_prompt(sample, demo)
        retry_count = 0  # Initialize retry counter

        while retry_count < 3:  # Allow up to 3 attempts (initial + 2 retries)
            try:
              messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": annotation_prompt}
    ]
              formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
              result=self.model(formatted_prompt,ParagraphLabels,max_new_tokens=10000,do_sample=False,use_cache=True)
              print(result)
              return self.postprocess(str(result))

            except Exception as e:
                print(f"Error during annotation: {e}")
                print(f"Problem was with: {annotation_prompt}")
                retry_count += 1  # Increment retry counter

                if retry_count == 3:
                    print("Max retries reached. Aborting operation.")
                    return None

                print("Retrying...")

        return None

    def postprocess(self, result):
        try:
          mach=re.search(r'\{\s*.*?\s*\}', result, re.DOTALL)
          if mach:
            result=mach.group(0)
          extracted_result=json.loads(result.strip())
        except json.JSONDecodeError:
          print("failed to parse JSON from result")
          return []
        outputs = []
        if isinstance(extracted_result, dict) and 'labels' in extracted_result:
            outputs.append(extracted_result)
        return outputs

class close_annotator:
    def __init__(self, engine: str = 'claude',use_demo: bool=True):
        self.input_format = input_format
        self.output_format = output_format
        self.engine=engine
        self.use_demo=use_demo
        self.n_shots = 5
        self.gemini = "gemini-2.0-flash-Lite"
        self.claude = "claude-haiku-4-5-20251001"
        if engine=="gemini":
          import getpass
          GEMINI_API_KEY = getpass.getpass("Enter your Gemini API key: ")
          if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite',system_instruction=system_template)
        if engine=="claude":
          self.client = anthropic.Anthropic(api_key="")


    def generate_prompt(self, sample, demo=None):
        to_annotate = self.input_format.format(json.dumps(sample['text']))
        if self.use_demo and demo:
            demo_file = {str(i['id']):i for i in demo}
            demo = [demo_file[str(pointer['id'])] for pointer in reversed(demo_index[str(sample['id'])])]
            demo_annotations = "\n".join(f"{{{self.input_format.format(json.dumps(d['text']))},{self.output_format.format(json.dumps(d['label']))}}}" for d in demo)
            return f"Here are some examples to guide the labeling:\n{demo_annotations}\n Now please annotate the following paragraph using topic labels:\n{to_annotate}"
        else:
            return f"please annotate the following paragraph using topic labels:\n{to_annotate}"

    @func_set_timeout(60)
    def online_annotate(self, sample, demo=None):
        annotation_prompt = self.generate_prompt(sample, demo)
        retry_count = 0  # Initialize retry counter

        while retry_count < 3:
            try:
              if self.engine == "gemini":
                response = self.model.generate_content(
                   annotation_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=ParagraphLabels.model_json_schema()
            )
        )
                decoded = response.text
                print(decoded)
                return self.postprocess(decoded)

              elif self.engine == "claude":
                  response = self.client.messages.parse(
                      model=self.claude,
                      max_tokens=1000,
                      temperature=0.0,
                      system=system_template,
                      messages=[{"role": "user", "content": annotation_prompt}],
                      output_format=ParagraphLabels,
                     )
                  parsed=response.parsed_output
                  print(parsed)
                  return [parsed.model_dump()] if parsed and hasattr(parsed, 'labels') else []
            except Exception as e:
                print(f"Error during annotation: {e}")
                print(f"Problem was with: {annotation_prompt}")
                retry_count += 1  # Increment retry counter

                if retry_count == 3:
                    print("Max retries reached. Aborting operation.")
                    return None

                print("Retrying...")

        return None

    def postprocess(self, result):
        try:
          mach=re.search(r'\{.*\}', result, re.DOTALL)
          if mach:
            result=mach.group(0)
          extracted_result=json.loads(result.strip())
        except json.JSONDecodeError:
          print("failed to parse JSON from result")
          return []
        outputs = []
        if isinstance(extracted_result, dict) and 'labels' in extracted_result:
            outputs.append(extracted_result)
        return outputs

class Annotator:
  def __init__(self, engine:str='claude',annotator_type:str='close'):
    self.annotator_type=annotator_type
    if annotator_type=='open':
      self.annotator=open_annotator(engine)
    elif annotator_type=='close':
      self.annotator=close_annotator(engine)
  def online_annotate(self,sample:dict,demo):
    if self.annotator_type=='open':
      return self.annotator.online_annotate(sample,demo)
    elif self.annotator_type=='close':
      return self.annotator.online_annotate(sample,demo)
annotator= Annotator(engine='mistral',annotator_type='open')

results = []
for sample in test:
    result = annotator.online_annotate(sample,demo)
    results.append(result)
