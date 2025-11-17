"""
annotates the training samples with the help of selected genAI with demo.

"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
from func_timeout import func_set_timeout
import re
import ujson as json
import google.generativeai as genai
import anthropic

class open_annotator:
    def __init__(self, engine: str=''):
        self.input_format = input_format
        self.output_format = output_format
        self.qwen = "Qwen/Qwen2-7B-Instruct"
        self.mistral = "mistralai/Mistral-7B-Instruct-v0.3"
        if engine=="qwen":
          self.tokenizer = AutoTokenizer.from_pretrained(self.qwen)
          self.model = AutoModelForCausalLM.from_pretrained(self.qwen,torch_dtype=torch.bfloat16,device_map="auto")
        elif engine=="mistral":
           self.tokenizer = AutoTokenizer.from_pretrained(self.mistral)
           self.model = AutoModelForCausalLM.from_pretrained(self.mistral,torch_dtype=torch.bfloat16,device_map="auto")
        if self.tokenizer.pad_token_id is None:
          self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_prompt(self, sample, demo=None):
        to_annotate = self.input_format.format(json.dumps(sample['text']))
        if demo:
            demo_annotations = "\n".join(
                f"{self.input_format.format(json.dumps(d['text']))}\n{self.output_format.format(json.dumps(d['label']))}" for d in demo
            )
            return f"Here are some examples to guide the labeling:\n{demo_annotations}\n\n Now annotate the following input:\n{to_annotate}"
        else:
            return f"Please annotate the following input:\n{to_annotate}"

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
              model_inputs = self.tokenizer(formatted_prompt, return_tensors="pt",padding=True).to("cuda")
              generated_ids = self.model.generate(model_inputs.input_ids,attention_mask=model_inputs.attention_mask,max_new_tokens=1000, do_sample=False)
              new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]
              decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
              print(decoded)
              return self.postprocess(str(decoded))

            except Exception as e:
                print(f"Error during annotation: {e}")
                print(f"Problem was with: {annotation_prompt}")
                retry_count += 1  # Increment retry counter

                if retry_count == 3:
                    print("Max retries reached. Aborting operation.")
                    return None

                print("Retrying")

        return None

    def postprocess(self, result):
        tagset = domain
        try:
          extracted_result=json.loads(result.strip())
          if not isinstance(extracted_result, list):
            extracted_result = [extracted_result]
        except json.JSONDecodeError:
          print("failed to parse JSON from result")
          return []
        outputs = []
        for entity in extracted_result:
            if not isinstance(entity, dict):
                continue
            if 'labels' not in entity:
                continue
            if all(label in tagset for label in entity):
                outputs.append(entity)
        return outputs

class close_Annotator:
    def __init__(self, engine: str=''):
        self.input_format = input_format
        self.output_format = output_format
        self.gemini = "gemini-1.5-pro-002"
        self.claude = "claude-3-7-sonnet-20250219"
        if engine=="gemini":
          import getpass
          GEMINI_API_KEY = getpass.getpass("Enter your Gemini API key: ")
          if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-pro-002',system_instruction=system_template)
        if engine=="claude":
          self.client = anthropic.Anthropic(api_key="")
    def generate_prompt(self, sample, demo=None):
        to_annotate = self.input_format.format(json.dumps(sample['text']))
        if demo:
            demo_annotations = "\n".join(
                f"{self.input_format.format(json.dumps(d['text']))}\n{self.output_format.format(json.dumps(d['labels']))}" for d in demo
            )
            return f"Here are some examples:\n{demo_annotations}\n\nNow annotate the following input:\n{to_annotate}"
        else:
            return f"Now annotate the following input:\n{to_annotate}"

    @func_set_timeout(60)
    def online_annotate(self, sample, engine, demo=None):
        annotation_prompt = self.generate_prompt(sample, demo)
        retry_count = 0  # Initialize retry counter

        while retry_count < 3:
            try:
              if engine == "gemini":
                response = self.gemini.generate_content(
                   annotation_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.0,
            )
        )

                decoded = response.text
                return self.postprocess(response)
              else:
                if engine == "claude":
                  response = self.client.messages.create(
                      model=self.claude,
                      max_tokens=1000,
                      temperature=0.0,
                      system=system_template,
                      messages=[{"role": "system", "content": annotation_prompt}]
                     )
            except Exception as e:
                print(f"Error during annotation: {e}")
                print(f"Problem was with: {annotation_prompt}")
                retry_count += 1  # Increment retry counter

                if retry_count == 3:
                    print("Max retries reached. Aborting operation.")
                    return None

                print("Retrying")

        return None

    def postprocess(self, result):
        tagset = domain
        try:
          extracted_result=json.loads(result.strip())
          if not isinstance(extracted_result, list):
            extracted_result = [extracted_result]
        except json.JSONDecodeError:
          print("failed to parse JSON from result")
          return []
        outputs = []
        for entity in extracted_result:
            if not isinstance(entity, dict):
                continue
            if 'labels' not in entity:
                continue
            if all(label in tagset for label in entity):
                outputs.append(entity)
        return outputs

class Annotator:
  def __init__(self, engine:str,annotator_type:str=''):
    self.annotator_type=annotator_type
    if annotator_type=='open':
      self.annotator=open_annotator(engine)
    elif annotator_type=='close':
      self.annotator=close_annotator(engine)
  def online_annotate(self,sample,demo,engine=None):
    if self.annotator_type=='open':
      return self.annotator.online_annotate(sample,demo)
    elif self.annotator_type=='close':
      return self.annotator.online_annotate(sample,engine,demo)
