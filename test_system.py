#pip install -r requirements.txt
from common_utils import JokeChatSystem
import time
import json
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
    try:
        system = JokeChatSystem('test_config.json')
    except FileNotFoundError:
        system = JokeChatSystem()
    
    system.load_models()

    test_cases: List[Dict[str, str]] = [
        {
            "dialogue": """
            John: I've been trying to learn programming but it's so frustrating.
            Mary: What specifically are you finding difficult?
            John: Everything! The syntax, the logic, debugging... it's overwhelming.
            Mary: Everyone feels that way at first. Maybe we could study together?
            John: Really? That would be great! Thanks for offering to help.
            """,
            "reference_summary": "John is frustrated with learning programming. Mary offers to study together to help him."
        },
        {
            "dialogue": """
            Tom: How was your vacation in Hawaii?
            Sarah: It was amazing! The beaches were beautiful and the food was incredible.
            Tom: Did you try surfing?
            Sarah: Yes! I took a lesson and managed to stand up on the board.
            Tom: That's awesome! I've always wanted to learn.
            Sarah: You should definitely try it. The instructors are really patient.
            """,
            "reference_summary": "Sarah shares her positive experience of her Hawaii vacation, including surfing lessons. Tom expresses interest in learning to surf."
        },
        {
            "dialogue": """
            Alex: Did you hear about the new coffee shop downtown?
            Emma: No, what's special about it?
            Alex: They have this unique brewing method and amazing pastries.
            Emma: Sounds interesting! Want to check it out tomorrow?
            Alex: Sure, let's meet there around 10?
            Emma: Perfect, see you then!
            """,
            "reference_summary": "Alex tells Emma about a new coffee shop downtown. They plan to visit it together tomorrow at 10."
        }
    ]

    print("\nRunning dialogue summary tests...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print("Original dialogue:")
        print(test_case["dialogue"])
        
        generated_summary = test_dialogue_summary(system, test_case["dialogue"])
        print(f"\nGenerated summary: {generated_summary}")
        print(f"Reference summary: {test_case['reference_summary']}")
        
        jokes = test_joke_recommendation(system, generated_summary)
        print("\nGenerated jokes based on summary:")
        for j, joke in enumerate(jokes, 1):
            print(f"Joke {j}: {joke}")

    print("\nTesting specific contexts...")
    specific_contexts = [
        "Someone struggling with computer programming",
        "A beautiful day at the beach",
        "A person learning to cook for the first time"
    ]
    
    for context in specific_contexts:
        print(f"\nContext: {context}")
        jokes = test_joke_recommendation(system, context)
        print("Generated jokes:")
        for i, joke in enumerate(jokes, 1):
            print(f"Joke {i}: {joke}")

if __name__ == "__main__":
    main()