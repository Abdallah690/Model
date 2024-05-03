---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- bleu
model-index:
- name: Hieroglyph-Translator-Using-Gardiner-Codes
  results: []
---

# Hieroglyph-Translator-Using-Gardiner-Codes

This model was created to translate hieroglyphs into english.

Egyptian Hieroglyphs have been grouped into different classes and given a referencing method called [Gardiner Codes](https://www.egyptianhieroglyphs.net/gardiners-sign-list/) using Gardiner Classification.

Using the Gardiner Codes we can assign meanings to different combinations of hieroglyphs.

To Translate any sequence of hieroglyphs using this model, provide the following input :-

"Translate hieroglyph gardiner code sequence to English: {Gardiner Codes of the Hieroglyphs}"

Examples :

"Translate hieroglyph gardiner code sequence to English: A4 A5 A1 B6 F8"

"Translate hieroglyph gardiner code sequence to English: A4 A5 G4 H9 P3"

It achieves the following results on the evaluation set:
- Loss: 3.4556
- Bleu: 0.4084
- Gen Len: 5.795



# Model description

This model is a fine-tuned version of t5-small on a custom dataset derived from the [Dictionary of Middle Egyptian](https://www.academia.edu/42457720/Dictionary_of_Middle_Egyptian_in_Gardiner_Classification_Order).

The Inference Api on the hugging face model page doesn't work well, load the model in jupyter notebook using the following code snippet:
  
    
    text = "Translate hieroglyph gardiner code sequence to English: A4 A5 "

    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Hieroglyph-Translator-Using-Gardiner-Codes")
    inputs = tokenizer(text, return_tensors="pt").input_ids
    
    from transformers import AutoModelForSeq2SeqLM
    
    model = AutoModelForSeq2SeqLM.from_pretrained("Hieroglyph-Translator-Using-Gardiner-Codes")
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    translated_keywords = str(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print(translated_keywords)



# Intended uses & limitations

The Model is intended to be used to translate hieroglyphs. The model does not provide full sentences, it only outputs bits and keywords.



### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step   | Validation Loss | Bleu   | Gen Len |
|:-------------:|:-----:|:------:|:---------------:|:------:|:-------:|
| 4.3013        | 1.0   | 11000  | 4.1166          | 0.2832 | 6.967   |
| 4.1299        | 2.0   | 22000  | 3.9282          | 0.5713 | 6.866   |
| 3.9448        | 3.0   | 33000  | 3.7724          | 0.1969 | 5.585   |
| 3.7424        | 4.0   | 44000  | 3.6706          | 0.4691 | 5.736   |
| 3.6359        | 5.0   | 55000  | 3.6008          | 0.2859 | 5.631   |
| 3.6102        | 6.0   | 66000  | 3.5475          | 0.338  | 5.722   |
| 3.4461        | 7.0   | 77000  | 3.5068          | 0.306  | 5.74    |
| 3.4753        | 8.0   | 88000  | 3.4755          | 0.4031 | 5.78    |
| 3.4109        | 9.0   | 99000  | 3.4567          | 0.4635 | 5.765   |
| 3.3798        | 10.0  | 110000 | 3.4556          | 0.4084 | 5.795   |


### Framework versions

- Transformers 4.27.4
- Pytorch 2.2.0.dev20231113
- Datasets 2.12.0
- Tokenizers 0.13.3
