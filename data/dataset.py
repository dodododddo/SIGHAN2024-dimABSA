import json
from datasets import Dataset

INTENSITY_MAPPING = []
  
def build_dataset(split="train"):
    data_path = None
    if split == "train":
        data_path = "./data/task2&3/train.json"
    elif split == "eval":
        data_path = "./data/task2&3/dev.json"
    else:
        raise ValueError("仅支持train/eval")
    
    data = [preprocess(x) for x in load_json(data_path)]
    
    data_dict = {"input_text": [x["input_text"] for x in data], "target_text": [x["target_text"] for x in data]}
    dataset = Dataset.from_dict(data_dict)
    return dataset
    
    
def preprocess(data:dict):
    src = data["Sentence"]
    aspects = data["Aspect"]
    categorys = data["Category"]
    opinions = data["Opinion"]
    intensities_origin = data["Intensity"]
    intensities = []
    for intensity_origin in intensities_origin:
        intensity = "#".join([str(round(float(x))) for x in intensity_origin.split("#")])
        intensities.append(intensity)
    
    
    dst = str(list(zip(aspects, opinions, categorys, intensities)))
    return {"input_text": src, "target_text": dst}
    

def load_json(path:str):
    with open(path, mode='r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return data
    

if __name__ == "__main__":
    dataset_train = build_dataset("train")
    print(dataset_train[0])

