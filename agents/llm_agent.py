import openai

class LLMAgent:
    def __init__(self, config):
        self.config = config
        self.provider = config["llm_api"]["provider"]
        self.model = config["llm_api"]["model"]
        self.temperature = config["llm_api"]["temperature"]
        self.max_tokens = config["llm_api"]["max_tokens"]
        
    def _call_llm_api(self, prompt):
        if self.provider == "openai":
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        
    def reflect_on_result(self, query, result, iteration):
        prompt = f"""
        Query: {query}
        Current Results: {result['results'][:5]}
        Uncertainty: {result['uncertainty']}
        Iteration: {iteration}
        
        Please analyze these results and suggest how to improve the query.
        """
        return self._call_llm_api(prompt)
        
    def refine_query(self, original_query, reflection, history):
        prompt = f"""
        Original Query: {original_query}
        Reflection: {reflection}
        Search History: {history}
        
        Please generate an improved version of the query based on the reflection.
        """
        return self._call_llm_api(prompt) 