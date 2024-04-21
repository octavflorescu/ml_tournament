import pandas as pd
import numpy as np
from collections import Counter
from thefuzz import process
import string

from huggingface_hub import login
login(token="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") # octavf's key


class TokenizerWrapper():
  def __init__(self, tokenizer, model):
    self.tokenizer = tokenizer
    self.model = model

  def naics3_tokenization(self, in_text):
    tokens = [str(a) for a in self.tokenizer([in_text], return_tensors="np")['input_ids'][0]]
    return "|".join([str(a) for a in tokens])

  def naics3_keywords_tokenization(self, in_text):
    query = f"{in_text}. The keywords are:"
    model_inputs = self.tokenizer([query], return_tensors="pt").to("cuda")
    generated_ids = self.model.generate(**model_inputs, max_new_tokens=50)
    out_keywords = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0][len(query):]
    out_keywords_tok = self.naics3_tokenization(out_keywords.replace('\n', '').replace('-', ' |'))
    return out_keywords_tok
  
  def expand(self, in_text, return_tokens=False):
    query = f"{in_text}, its main focus activity being"
    model_inputs = self.tokenizer([query], return_tensors="pt").to("cuda")
    generated_ids = self.model.generate(**model_inputs, max_new_tokens=12)
    if return_tokens:
      return generated_ids[0].detach().cpu().numpy()
    return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0][len(query):]

class NAICSClassifier:
    def __init__(self, tokenizer: TokenizerWrapper, DATA_PATH='/content/ml_tournament/resources/', naics_categories_path=None, naics_df:pd.DataFrame=None):
        self.naics3_target = pd.read_csv(f"{DATA_PATH}/naics3_augmented.csv" if naics_categories_path is None else naics_categories_path) if naics_df is None else naics_df
        self.naics3_choices = self.naics3_target.naics_label.tolist()
        self.toktok = tokenizer

    def classify(self, hint, complexity=2):
        """
        complexity 1 for round 1, company name: use only untokenized features 
        any complexity above, will use more features. e.g. tokenized summary
        """
        hints = [
            hint
            # sample0.commercial_name,
            # ','.join(sample0.business_tags.split(' |')),
            # sample0.short_description,
            # sample0.description,
            # sample0.main_business_category
        ]
        return self.classify_list_of_hints(hints, complexity=complexity)
        
    def classify_list_of_hints(self, hints, complexity=2):
        tokenized_sample_features = [*self.toktok.tokenizer(hints, return_tensors="np")['input_ids'].tolist()]
        summary_features = [self.toktok.expand(feature, return_tokens=False).translate(str.maketrans('', '', string.punctuation)).split() # strip punctuation
                            for feature in hints]
        naics3_confidences = [0] * len(self.naics3_choices)

        for idx, (feature, summary_feature, tokenized_feature) in enumerate(zip(hints, summary_features, tokenized_sample_features)):    
            for wrd in summary_feature:
                for idx, row in self.naics3_target.iterrows():
                    if wrd.lower() in row['naics_label'].lower():
                        naics3_confidences[idx] += 0.25 # if complexity > 1 else 25
                    if 'subcategories' in row and (wrd.lower() in row['subcategories'].lower()):
                        naics3_confidences[idx] += 0.15
                    if wrd.lower() in row['description'].lower():
                        naics3_confidences[idx] += 0.1
                
            choisez = process.extract(feature, self.naics3_choices, limit=5)  # => [('Health and Personal Care Retailers', 55), ('Water Transportation', 52), ('Real Estate', 51), ('Gasoline Stations and Fuel Dealers', 48), ('Specialty Trade Contractors', 47)]
            for choise, choise_score in choisez:
                idx = np.where(self.naics3_target["naics_label"] == choise)[0][0]
                naics3_confidences[idx] += choise_score*0.01
            
            if complexity > 1:
                for tok, counts in Counter(tokenized_feature).items():
                    for idx, row in self.naics3_target.iterrows():
                        if str(tok) in Counter(row['Mistral7Bv01_tokenized_name'].split("|")).keys():
                            naics3_confidences[idx] += 0.25
                        if 'Mistral7Bv01_tokenized_subcats' in row and (wrd.lower() in row['Mistral7Bv01_tokenized_subcats'].lower()):
                            naics3_confidences[idx] += 0.15
                        if str(tok) in Counter(row['Mistral7Bv01_tokenized_description'].split("|")).keys():
                            naics3_confidences[idx] += 0.1

        best_match_naics3_counts = 0
        for idx, current_naics3_match_counts in enumerate(naics3_confidences):
            if current_naics3_match_counts > best_match_naics3_counts:
                best_match_naics3_counts = current_naics3_match_counts

        return naics3_confidences
