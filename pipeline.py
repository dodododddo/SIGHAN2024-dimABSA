from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Trainer,
    pipeline, TrainingArguments
)
from data.dataset import build_dataset
import argparse
import ast
from tqdm import tqdm

def encoder_gen(tokenizer):
    def encode(examples):
            inputs = examples["input_text"]
            targets = examples["target_text"]
            input_encodings = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
            target_encodings = tokenizer(targets, truncation=True, padding="max_length", max_length=512)
            return {"input_ids": input_encodings.input_ids, "attention_mask": input_encodings.attention_mask, "labels": target_encodings.input_ids}
    return encode


def train():
    tokenizer = AutoTokenizer.from_pretrained("/home/chenjiarui/SIGHAN_GEN/model/t5_base_chinese", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("/home/chenjiarui/SIGHAN_GEN/model/t5_base_chinese", local_files_only=True)
    
    train_args = TrainingArguments(
        per_device_train_batch_size=2,
        num_train_epochs=10,
        logging_dir='./logs',
        output_dir='./outputs',
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        gradient_accumulation_steps=1
    )
    
    encode = encoder_gen(tokenizer)
    train_dataset = build_dataset("train").map(encode, batched=True)
    eval_dataset = build_dataset("eval").map(encode, batched=True)
    
    trainer = Trainer(model, train_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()
    
    
   
def eval():
    model = AutoModelForSeq2SeqLM.from_pretrained("./outputs/checkpoint-3000")
    tokenizer = AutoTokenizer.from_pretrained("./outputs/checkpoint-3000")
   
    pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)
    eval_dataset = build_dataset("eval")
    
    tp = 0
    tp_fp = 0
    tp_fn = 0
    for eval_data in tqdm(eval_dataset):
        ground_truth = ast.literal_eval(eval_data['target_text'])
        predict = ast.literal_eval(pipe(eval_data['input_text'], max_length=512)[0]['generated_text'])
        tp_fp += len(ground_truth)
        tp_fn += len(predict)
        for pred in predict:
            for gt in ground_truth:
                if pred == gt:
                    tp += 1
    
    precision = tp / tp_fp
    recall = tp / tp_fn
    f1 = 2 * precision * recall / (precision + recall)
    
    print(f'precision: {precision}, recall: {recall}, f1: {f1}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training code')
    parser.add_argument('--eval', action='store_true', help='Run evaluation code')

    args = parser.parse_args()
    if args.train:
        train()
        
    if args.eval:
        eval()
        
    if(not args.train and not args.eval):
        eval()
    

