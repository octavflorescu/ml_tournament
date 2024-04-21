import requests


class APIClient:

    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "x-api-key": self.api_key
        }

    def reset_current_context(self):
        response = requests.get("{0}/evaluate/reset".format(self.api_url), headers=self.headers)
        print(response.status_code, response.json())
        split_response = response.json()['response'].split(' ')
        return float(split_response[-2] if len(split_response) > 1 else 0)

    def send_answer_for_current_company(self, prediction):
        answer_object = {
            "answer": prediction
        }
        print("Answered", answer_object)
        response = requests.post("{0}/evaluate/answer".format(self.api_url), json=answer_object, headers=self.headers)
        print("Verdict", response.status_code, response.json())

        return response.json()

    # Get a new hint for current company.
    # Get the first hint for a new company after calling /evaluate/reset.
    def get_next_hint_for_current_company(self):
        response = requests.get("{0}/evaluate/hint".format(self.api_url), headers=self.headers, timeout=None)
        print(response.status_code, response.json())
        return response.json()
