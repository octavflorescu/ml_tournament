import time

from classifier import NAICSClassifier


class Evaluator:

    def __init__(self, api_url, api_key, api_client):
        self.api_url = api_url
        self.api_key = api_key
        self.classifier = NAICSClassifier()
        self.api_client = api_client
        self.hints = []
        self.scores = []

    def evaluate(self):
        # Round 1
        self.do_round()
        self.do_round()
        self.do_round()
        self.do_round()
        self.do_round()
        self.do_round()
        self.api_client.reset_current_context()

    def do_round(self):
        hint = self.api_client.get_next_hint_for_current_company()
        self.hints.append(hint)
        time.sleep(1)
        verdict = self.api_client.send_answer_for_current_company(self.classifier.classify(hint))
        self.scores.append(verdict["score"])
        time.sleep(1)
