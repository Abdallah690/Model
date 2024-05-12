#!/usr/bin/env python
# coding: utf-8

# In[4]:


# --For linux 
#%pip install transformers huggingface_hub 
#%pip install  huggingface_hub
#%pip install pandas
#%pip install flask
#%pip install torch torchvision torchaudio
# -- For Windows 
get_ipython().system('pip install transformers huggingface_hub')
get_ipython().system('pip install  huggingface_hub')
get_ipython().system('pip install torch torchvision torchaudio')
get_ipython().system('pip install pandas')
get_ipython().system('pip install flask')


# In[5]:


# Import the classes
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torchvision
from flask import Flask, request, jsonify
import pandas as pd
import logging


# In[6]:


print(torch.__version__)
x = torch.rand(5, 3)
print(x)


# In[17]:


# Define the path to the directory containing tokenizer and model files
# model_path = "/Users/genesis/Projects/Abdallah_Valo/Model-main/Model"

# # Load the tokenizer and the model from the custom path
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
# Load model directly
# Define the path to the directory containing tokenizer and model files
model_path = "F:\Model"

# Load the tokenizer and the model from the custom path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# In[19]:


# تحديد مسار ملف CSV
file_path = "F:\Model\HHH.csv"

# قراءة ملف CSV
df = pd.read_csv(file_path)

# افترض أن العمود الأول يحتوي على الرموز الهيروغليفية والثاني على الترجمة الإنجليزية
translation_dict = pd.Series(df['english'].values, index=df['hieroglyphs']).to_dict()


def translate_hieroglyphs(hieroglyphs):
    # تقسيم النص إلى رموز فردية إذا لزم الأمر
    symbols = hieroglyphs.split()
    # استخدام القاموس للترجمة
    translation = ' '.join(translation_dict.get(symbol, 'not found') for symbol in symbols)
    return translation
#𓄿
# استخدام الوظيفة
hieroglyphic_text = "𓀆"  # استبدل بالرموز الهيروغليفية الخاصة بك
#hieroglyphic_text = input("enter your sympole : ")  # استبدل بالرموز الهيروغليفية الخاصة بك
english_translation = translate_hieroglyphs(hieroglyphic_text)
print(english_translation)


# # Setup Logging

# In[10]:


# Setup logging
logging.basicConfig(level=logging.INFO)


# # Load Model And Tokenizer

# In[18]:


# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    logging.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model or tokenizer: {e}")
    raise SystemExit(e)


# # Load Translation Dictionary

# In[13]:


# Load translation dictionary
try:
    file_path = "F:\Model\HHH.csv"
    df = pd.read_csv(file_path)
    translation_dict = pd.Series(df['english'].values, index=df['hieroglyphs']).to_dict()
    logging.info("Translation dictionary loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load translation dictionary: {e}")
    raise SystemExit(e)


# # Define Translation Function
# 

# In[14]:


def translate_hieroglyphs(hieroglyphs):
    symbols = hieroglyphs.split()
    translation = ' '.join(translation_dict.get(symbol, 'not found') for symbol in symbols)
    return translation



# # Setup Flask App
# 

# In[15]:


#API End point
app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        hieroglyphs = data.get('hieroglyphs', '')
        translated_text = translate_hieroglyphs(hieroglyphs)
        return jsonify({'translated_text': translated_text})

    except Exception as e:
        logging.error(f"Error processing the translation request: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500


# # Run Flask App

# In[ ]:


if __name__ == '__main__':
    app.run(debug=False, port=5002)


# In[ ]:


import torch
import torchvision.models as models

# افترض أن model هو النموذج الذي تم تدريبه بالفعل
model_path = "F:\\Model"
torch.save(model_path, 'model05.pt')


# In[ ]:




