import random
from datasets import Dataset


## Doing things manually

# prompt_format = """
# <|start_of_text|>
# <|start_header_id|>system<|end_header_id|>
# You are a medical AI assistant for diagnose whether the patient has diabetes mellitus given their discharge notes. You will provide a single yes/no as the answer.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# Here is a patient's discharge summary report from a hospital encounter. Provide a one word yes or no answer to the following question: does the patient have diabetes mellitus?
# *Discharge summary report starts*
# {}
# *Discharge summary report ends*<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# {}<|eot_id|>
# <|end_of_text|>
# """

# def formatting_prompts_func(examples):
#     notes  =  examples["TEXT"]
#     codes = examples["CODES"]
#     texts = []
#     for note, code in zip(notes, codes):
#       answer = None
#       for each in code.split(','):
#         try:
#           value = int(each)
#           if value  >= 25000 and value <= 25099:
#             answer = 1
#             break
#           else:
#             continue
#         except ValueError:
#           continue
#       if answer == 1:
#         answer = "YES"   
#       else:
#         answer = "NO"
#       text = prompt_format.format(note, answer)
#       texts.append(text)
#     return { "training_text" : texts, }
# pass

# dataset = dataset.map(formatting_prompts_func, batched = True,)



def formatting_func(row, tokenizer):
  notes = row['TEXT']
  codes = row["CODES"]
  texts = []
  for note, code in zip(notes, codes):
    answer = check_diabetes(code)
    prompt = get_prompt(note, answer)
    texts.append(prompt)
  
  # encodeds = tokenizer.apply_chat_template(texts, return_tensors="pt", padding=True)
  text_field = tokenizer.apply_chat_template(texts, tokenize=False)

  return {"training_text": text_field,}


def check_diabetes(code):
    answer = None
    for each in code.split(','):
        try:
           value = int(each)
           if value  >= 25000 and value <= 25099:
             answer = 1
             break
           else:
             continue
        except ValueError:
           continue    
    if answer == 1:
      answer = "YES"   
    else:
      answer = "NO"
    return answer


def get_prompt(note, answer):
    prompt = []
    system_content = """You are a medical AI assistant for diagnose whether the patient has diabetes mellitus given their discharge notes. You will provide a single yes/no as the answer."""

    user_content = f"""Here is a patient's discharge summary report from a hospital encounter. Provide a one word yes or no answer to the following question: does the patient have diabetes mellitus?
    
    **DISCHARGE SUMMARY REPORT**
    **[REPORT START]**
    {note}
    **[REPORT END]**
    """

    assistant_content = answer

    system = {"role": "system", 
              "content": system_content}
    user = {"role": "user", 
              "content": user_content}
    assistant = {"role": "assistant", 
              "content": assistant_content}
    
    prompt.append(system)
    prompt.append(user)
    prompt.append(assistant)

    return prompt


def balance_data(dataset, delimiter="<|start_header_id|>assistant<|end_header_id|>"): 
    positive_samples = []
    negative_samples = []
    for sample in dataset:
       response = sample["training_text"].split(delimiter)[-1].strip()
       if "yes" in response.lower():
           positive_samples.append(sample)
       else:
           negative_samples.append(sample)
    num_positive_samples = len(positive_samples)

    # Randomly sample the same number of negative samples
    sampled_negative_samples = random.sample(negative_samples, num_positive_samples)
    balanced_samples = positive_samples + sampled_negative_samples
    print(f"Number of rows that report yes: {num_positive_samples}")
    print(f"Number of rows that report no: {num_positive_samples}")

    # Convert the list of balanced samples back to a Dataset
    balanced_dataset = Dataset.from_dict({key: [sample[key] for sample in balanced_samples] for key in balanced_samples[0]})
    return balanced_dataset