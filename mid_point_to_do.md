# Mid point update


## What has been done: 

- LoRA finetuning of tinyLM
  - in the notebook file title `Lora_trained_vs_untrained_embeddings.ipynb` LoRA is used to finetune TinyLM llm
  - As the title suggests this has been done to both finetuned embeddings and not finetuned embeddings with results plotted indicating better performance for the finetuned embeddings
- Finetuning of word embeddings: 
  - in the notebook file entitled `train_embeddings.ipynb` and the file `train_embeddings_all_subjects`, you can see the training process for the word embeddings based on a pretrained model from the sentence transformers library


## What should be done going forward: 

- Train on Sam's custom dataset
  - use Sam's custom dataset which is way larger than wildgraph in order to train the word embeddings and to use as training data for LoRA finetuning of the LLM itself. 
  
- Repeat everything that has been done on larger and more advanced models.
  - It would be nice to get as much performance out of tinyLM as possible, but the reality is models like mistral 7B will perform better. Finetuning these larger models will likely lead to the best possible performance. 
  - Most of this can be done by copying the existing notebooks and swapping in different models. 
  - Training on the custom dataset will most likely require some tricky formatting but it will be worth it as the dataset is larger and will help the model generalize to different testing data. 
