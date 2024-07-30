from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login

## Login to Huggingface
login(token="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")

file_paths = ["Notes_ICD_0-5k.json", "Notes_ICD_5-10k.json", "Notes_ICD_15-20k.json", "Notes_ICD_25-30k.json", "Notes_ICD_30-50k.json", "Notes_ICD_50-70k.json", "Notes_ICD_70-100k.json"]
dataset_name = "wangrice/MIMIC_III_NotesICD_100k"
dataset = load_dataset(dataset_name, split="train", data_files=file_paths)

def map_function(row):
  codes = row["CODES"]
  answers = []
  for code in codes:
     answer = check_diabetes(code)
     answers.append(answer)

  return {"Diabetes_Diagnosis": answers}
     

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

dataset = dataset.map(map_function, batched=True)
dataset.push_to_hub("wangrice/MIMIC_III_NotesICD_100k_diabetes", token = "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")
