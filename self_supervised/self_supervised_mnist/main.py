# main.py

import sys
sys.path.append('.')

import train.train as train
import finetune.finetune as finetune

def main():
    # Step 1: Train the model on the rotation prediction task
    print("Training model on pretext task (rotation prediction)...")
    train.train_model()
    
    # Step 2: Fine-tune the model on digit classification
    print("Fine-tuning model on digit classification...")
    finetune.fine_tune_and_evaluate()

if __name__ == "__main__":
    main()
