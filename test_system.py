from common_utils import JokeChatSystem
import evaluate
import time

def test_dialogue_summary(system, dialogue):
    """
    대화 요약 기능을 테스트하는 함수입니다.
    
    Args:
        system: JokeChatSystem 인스턴스
        dialogue: 테스트할 대화 문자열
    
    Returns:
        생성된 요약문
    """
    print("\n=== Testing Dialogue Summary ===")
    print("Original dialogue:")
    print(dialogue)
    
    # 시간 측정 시작
    start_time = time.time()
    
    summary = system.generate_summary(dialogue)
    
    # 실행 시간 계산
    execution_time = time.time() - start_time
    
    print("\nGenerated Summary:")
    print(summary)
    print(f"\nSummary generation took {execution_time:.2f} seconds")
    
    return summary

def test_joke_recommendation(system, context):
    """
    농담 추천 기능을 테스트하는 함수입니다.
    
    Args:
        system: JokeChatSystem 인스턴스
        context: 농담 생성을 위한 컨텍스트 문자열
    
    Returns:
        생성된 농담 리스트
    """
    print("\n=== Testing Joke Recommendation ===")
    print("Context for joke generation:")
    print(context)
    
    # 시간 측정 시작
    start_time = time.time()
    
    jokes = system.recommend_joke(context)
    
    # 실행 시간 계산
    execution_time = time.time() - start_time
    
    print("\nRecommended Jokes:")
    for i, joke in enumerate(jokes, 1):
        print(f"{i}. {joke}")
    print(f"\nJoke generation took {execution_time:.2f} seconds")
    
    return jokes

def evaluate_summary_quality(system, dialogue, generated_summary, reference_summary):
    """
    생성된 요약의 품질을 평가하는 함수입니다.
    
    Args:
        system: JokeChatSystem 인스턴스
        dialogue: 원본 대화
        generated_summary: 생성된 요약
        reference_summary: 참조 요약
    
    Returns:
        ROUGE 점수 결과
    """
    print("\n=== Evaluating Summary Quality ===")
    print("Reference summary:", reference_summary)
    print("Generated summary:", generated_summary)
    
    results = system.evaluate_summary(dialogue, generated_summary, reference_summary)
    
    print("\nROUGE Scores:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
    return results

def main():
    """
    전체 시스템을 테스트하는 메인 함수입니다.
    다양한 대화 시나리오에 대해 시스템의 성능을 테스트합니다.
    """
    print("Initializing JokeChatSystem...")
    system = JokeChatSystem(model_dir='saved_models')
    
    print("Loading trained models...")
    system.load_models()

    # 다양한 상황의 테스트 대화 샘플들
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

    # 각 테스트 케이스에 대해 시스템 테스트 수행
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test Case {i}")
        print(f"{'='*50}")
        
        # 요약 생성 테스트
        generated_summary = test_dialogue_summary(system, test_case["dialogue"])
        
        # 요약 품질 평가
        if "reference_summary" in test_case:
            evaluate_summary_quality(
                system,
                test_case["dialogue"],
                generated_summary,
                test_case["reference_summary"]
            )
        
        # 생성된 요약을 기반으로 농담 추천 테스트
        test_joke_recommendation(system, generated_summary)
        
        # 테스트 케이스 간 구분선 추가
        print("\n" + "="*50)

    # 특정 상황에 대한 직접적인 농담 생성 테스트
    print("\nTesting direct joke generation for specific contexts...")
    specific_contexts = [
        "Someone struggling with computer programming",
        "A beautiful day at the beach",
        "A person learning to cook for the first time"
    ]
    
    for context in specific_contexts:
        test_joke_recommendation(system, context)

if __name__ == "__main__":
    main()