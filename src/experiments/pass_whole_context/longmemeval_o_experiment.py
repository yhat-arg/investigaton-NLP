from src.models.LiteLLMModel import LiteLLMModel
from src.agents.judge.JudgeAgent import JudgeAgent
from src.agents.full_context.FullContextAgent import FullContextAgent
from src.core.LongMemEvalDataset import LongMemEvalDataset
from src.models.TransformersModel import TransformersModel


def run_experiment(N: int):
    # litellm_model = LiteLLMModel("azure/gpt-4.1")
    model = TransformersModel("Qwen/Qwen3-1.7B")
    judge_agent = JudgeAgent(model)
    memory_agent = FullContextAgent(model)

    correct_predictions = 0
    longmemeval_o_dataset = LongMemEvalDataset("oracle")

    for question, sessions, t_question, answer in longmemeval_o_dataset[:N]:
        predicted_answer = memory_agent.answer(sessions, question, t_question)
        answer_is_correct = judge_agent.judge(question, predicted_answer, answer)
        if answer_is_correct:
            correct_predictions += 1

        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Answer: {answer}")
        print(f"Correct: {answer_is_correct}")
        print("-" * 100)

    print(f"Correct predictions: {correct_predictions}/{N}")
    print(f"Accuracy: {correct_predictions/N}")
