from common_utils import JokeChatSystem
from config import SystemConfig
import evaluate
import time

def test_dialogue_summary(system, dialogue):
    summary = system.generate_summary(dialogue)
    return summary

def test_joke_recommendation(system, context):
    start_time = time.time()
    jokes = system.recommend_joke(context)
    execution_time = time.time() - start_time
    return jokes

def main():
    try:
        config = SystemConfig.load_config('test_config.json')
    except FileNotFoundError:
        config = SystemConfig()
    
    system = JokeChatSystem(config=config)
    system.load_models()

    test_cases = [
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

    for i, test_case in enumerate(test_cases, 1):
        generated_summary = test_dialogue_summary(system, test_case["dialogue"])
        test_joke_recommendation(system, generated_summary)

    specific_contexts = [
        "Someone struggling with computer programming",
        "A beautiful day at the beach",
        "A person learning to cook for the first time"
    ]
    
    for context in specific_contexts:
        test_joke_recommendation(system, context)

if __name__ == "__main__":
    main()