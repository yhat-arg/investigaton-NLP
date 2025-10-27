from src.models.LiteLLMModel import LiteLLMModel
from src.agents.judge.JudgeAgent import JudgeAgent
from src.agents.full_context.FullContextAgent import FullContextAgent
from src.core.LongMemEvalDataset import LongMemEvalDataset
from src.models.QwenModel import QwenModel


def run_experiment(config):
    if config["memory_model_type"] == "transformers":
        memory_model = QwenModel(
            config["memory_model_name"], quantized=config["memory_model_quantized"]
        )
    elif config["memory_model_type"] == "litellm":
        memory_model = LiteLLMModel(config["memory_model_name"])
    else:
        raise ValueError(f"Invalid model type: {config['memory_model_type']}")

    judge_model = LiteLLMModel(config["judge_model_name"])
    judge_agent = JudgeAgent(model=judge_model)
    memory_agent = FullContextAgent(model=memory_model)

    correct_predictions = 0
    longmemeval_o_dataset = LongMemEvalDataset(config["longmemeval_dataset_type"])

    for question, sessions, t_question, answer in longmemeval_o_dataset[: config["N"]]:
        predicted_answer = memory_agent.answer(sessions, question, t_question)
        answer_is_correct = judge_agent.judge(question, predicted_answer, answer)
        if answer_is_correct:
            correct_predictions += 1

        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Answer: {answer}")
        print(f"Correct: {answer_is_correct}")
        print("-" * 100)

    print(f"Correct predictions: {correct_predictions}/{config['N']}")
    print(f"Accuracy: {correct_predictions/config['N']}")
