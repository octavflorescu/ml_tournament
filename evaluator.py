import time

from classifier import NAICSClassifier

class GameRound():
    def __init__(self, hint=None, prediction=None, score=None):
        self.hint = hint
        self.prediction = prediction
        self.score = score

class Evaluator:

    def __init__(self, api_url, api_key, api_client):
        self.api_url = api_url
        self.api_key = api_key
        self.classifier = NAICSClassifier()
        self.api_client = api_client

    def evaluate(self): 
        rounds = []
        # Round 1
        rounds.append(self.do_round(1, sleep=True))
        # Round 2
        rounds.append(self.do_round(2, sleep=True))
        # Round 3
        rounds.append(self.do_round(3, sleep=True))
        # Round 4
        rounds.append(self.do_round(4, sleep=True))
        # Round 5
        rounds.append(self.do_round(5, sleep=True))
        self.api_client.reset_current_context()

    def do_round(self, round_id=1, sleep=False):
        hint = self.api_client.get_next_hint_for_current_company()
        game_round = GameRound(hint=hint)
        if sleep: time.sleep(1)
        game_round.prediction = self.classifier.classify(hint, round_id)
        verdict = self.api_client.send_answer_for_current_company()
        game_round.score = verdict["score"]
        if sleep: time.sleep(1)
        return game_round
