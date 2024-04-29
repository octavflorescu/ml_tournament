import time
import numpy as np

from classifier import NAICSClassifier, TokenizerWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer


class GameRound():
    def __init__(self, hint=None, predictions=None, score=None):
        self.hint = hint
        self.predictions = predictions
        self.answer = None
        self.score = score
        self.rounds = []

class Evaluator:

    def __init__(self, api_url, api_key, api_client,
                 cache_dir='/content/sample_data/',
                 DATA_PATH='/content/ml_tournament/resources/'):
        self.api_url = api_url
        self.api_key = api_key
        
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left", cache_dir=cache_dir)
        self.tokenizer = TokenizerWrapper(tokenizer, model)
        self.classifier = NAICSClassifier(tokenizer=self.tokenizer, DATA_PATH=DATA_PATH, naics_categories_path=f'{DATA_PATH}/NAICS3_with_merged_Subcategories.csv')
        self.api_client = api_client

    def evaluate(self, sleep=0.1): 
        self.rounds = []
        # Round 1
        self.rounds.append(self.do_round(round_id=1, sleep=sleep))
        # Round 2
        self.rounds.append(self.do_round(round_id=2, sleep=sleep))
        # Round 3
        self.rounds.append(self.do_round(round_id=3, sleep=sleep))
        # Round 4
        self.rounds.append(self.do_round(round_id=4, sleep=sleep))
        # Round 5
        self.rounds.append(self.do_round(round_id=5, sleep=sleep))
        self.api_client.reset_current_context()

    def do_round(self, round_id=1, sleep=0.1, enable_hack=False):
        reponse = self.api_client.get_next_hint_for_current_company()
        hint = reponse['hint']
        game_round = GameRound(hint=hint)
        if sleep: time.sleep(sleep)
        game_round.predictions = self.classifier.classify(hint, complexity=round_id)
        game_round.answer = self.extract_round_prediction(current_predictions=game_round.predictions, enable_hack=enable_hack)
        verdict = self.api_client.send_answer_for_current_company(prediction=game_round.answer)
        game_round.score = verdict["score"]
        if sleep: time.sleep(sleep)
        return game_round
    
    def compete(self, round_id=1, sleep=0.1):
        if round_id == 1:
            self.rounds = []
        # Round 1
        self.rounds.append(self.do_round(round_id=1, sleep=sleep))
        if round_id == 5:
            self.api_client.reset_current_context()

    def extract_round_prediction(self, current_predictions, round_id=1, enable_hack=False):
        if (round_id == 5) and enable_hack:
            return "abstain"
        naics3_scores = np.zeros(len(self.classifier.naics3_choices))
        for idx, round in enumerate(self.rounds):
            naics3_scores += np.array(round.predictions)
        naics3_scores += np.array(current_predictions)
        # print(sorted(zip(naics3_scores, self.classifier.naics3_choices), reverse=True)[:5])
        return self.classifier.naics3_choices[np.argmax(naics3_scores)]
