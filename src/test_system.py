#pip install -r requirements.txt
import warnings
warnings.filterwarnings("ignore")

from utils.common_utils import JokeChatSystem
import time
from typing import Dict, List, Any

def test_dialogue_summary(system: JokeChatSystem, dialogue: str) -> str:
    return system.generate_summary(dialogue)

def test_joke_recommendation(system: JokeChatSystem, context: str) -> List[str]:
    start_time = time.time()
    jokes = system.recommend_joke(context)
    execution_time = time.time() - start_time
    print(f"Joke generation time: {execution_time:.2f} seconds")
    return jokes

def main():
    system = JokeChatSystem()
    
    system.load_models(epoch = 'best')

    test_case = """
You: How’s the bookshelf you were building? Did it turn out okay?
Chris: Uh, not exactly.
You: What do you mean?
Chris: I followed the instructions perfectly, but now it leans more than the Tower of Pisa.
You: Oh no! Did you fix it?
Chris: Nope, I just called it “modern art” and left it as is.
            """

    print("Original dialogue:")
    print(test_case)
    
    generated_summary = test_dialogue_summary(system, test_case)
    print(f"\nGenerated summary: {generated_summary}")
    
    jokes = test_joke_recommendation(system, generated_summary)
    print("\nGenerated joke based on summary:")
    print(jokes)

if __name__ == "__main__":
    main()






"""
You: How’s the bookshelf you were building? Did it turn out okay?
Chris: Uh, not exactly.
You: What do you mean?
Chris: I followed the instructions perfectly, but now it leans more than the Tower of Pisa.
You: Oh no! Did you fix it?
Chris: Nope, I just called it “modern art” and left it as is.
"""

"""
You: How’s your new puppy doing?
Taylor: She’s great! But she chewed up my slippers yesterday.
You: Oh no, your favorite ones?
Taylor: Yep, and now she’s walking around the house like she owns the place.
You: Did she at least leave the other slipper intact?
Taylor: Nope, it’s gone too. Guess I’m stuck with cold feet for a while.
"""