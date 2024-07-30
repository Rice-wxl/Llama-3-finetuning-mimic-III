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



def formatting_func(row, tokenizer, field=None, is_testing=False):
  notes = row['TEXT']
  if is_testing:
    answers = check_diabetes(row['CODES'])
  else:
    answers = row[field]
  texts = []
  for note, answer in zip(notes, answers):
    prompt = get_prompt(note, answer, is_testing)
    texts.append(prompt)
  
  if is_testing:
    text_field = tokenizer.apply_chat_template(texts, tokenize=False, add_generation_prompt=True)
  else:
    text_field = tokenizer.apply_chat_template(texts, tokenize=False)

  return {"training_text": text_field,}


def check_diabetes(codes):
  answers = []
  for code in codes:
    answer = None
    for each in code.split(','):
        try:
           value = int(each)
           if value == 99591 or value == 99592 or value == 78552:
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
    answers.append(answer)

  return answers


def get_prompt(note, answer, is_testing):
    prompt = []
    system_content = """You are a medical AI assistant specializing in disease diagnosis. You will read a discharge summary report of a patient, and determine whether the patient has sepsis. Please provide a single yes/no answer."""

    user_content = f"""Based on the following discharge summary report of a patient, you will determine whether the patient has sepsis and output a single word, either YES or NO, as the answer. Sepsis is defined as a life-threatening condition that happens when the body's immune system has an extreme response to an infection. It can be either severe sepsis, which is with acute organ dysfunction, or non-severe sepsis. It also includes septic shock, where a bacterial infection causes low blood pressure.
    
    *discharge summary report starts*
    {note}
    *discharge summary report ends*

    Remember to output a single word, either YES or NO, as the answer. In particular, if sepsis is included as a diagnosis in the section: Discharge Diagnosis, you should output YES. Otherwise you need to look at other places in the note where it mentions sepsis. Remember that some notes might say the patient has presumed sepsis, but either do not confirm its presence or say that sepsis is ruled out. In that case you should output NO.
    """

    assistant_content = answer

    system = {"role": "system", 
              "content": system_content}
    user = {"role": "user", 
              "content": user_content}
    
    prompt.append(system)
    prompt.append(user)

    if is_testing == False:
      assistant = {"role": "assistant", 
                "content": assistant_content}
      
      prompt.append(assistant)

    return prompt


def balance_data(dataset, field): 
    positive_samples = []
    negative_samples = []
    for sample in dataset:
       response = sample[field]
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