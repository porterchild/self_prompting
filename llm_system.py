import os
import time
import re
import textwrap
from openai import OpenAI

# Configuration
NUM_ITERATIONS = 50
NUM_TRIALS = 10
MODEL = "llama3.1"  # Adjust if needed

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Initialize OpenAI client for local Ollama server
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:8081/v1"
)

# Static question and correct answer
static_question = "A juggler throws a solid blue ball a meter in the air and then a solid purple ball (of the same size) two meters in the air. She then climbs to the top of a tall ladder carefully, balancing a yellow balloon on her head. Where is the purple ball most likely now, in relation to the blue ball?"
correct_answer = "same height as the blue ball (they're both on the ground)"

def make_api_call(client, model, messages, max_retries=3):
    """
    Helper function to make API call with retries on connection errors.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response
        except Exception as e:
            if "Connection" in str(e) or "ReadError" in str(e):
                if attempt < max_retries - 1:
                    print(f"Connection error on attempt {attempt + 1}, retrying in 2 seconds...")
                    time.sleep(2)
                    continue
            raise e
    raise Exception(f"Failed after {max_retries} retries due to connection issues.")

def question_answerer(client, prompt, question):
    """
    Question Answerer: Generates an answer using the provided prompt and question.
    """
    full_prompt = f"{prompt}\n\nQuestion: {question}"
    response = make_api_call(
        client, MODEL, [{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content.strip()

def supervisor_evaluate(client, answers, correct_answer):
    """
    Supervisor Evaluation: Runs binary correct/incorrect judgment on each answer using LLM.
    Returns accuracy as fraction (0 to 1).
    """
    correct_count = 0
    for i, answer in enumerate(answers, 1):
        if not answer:
            print(f"Trial {i} failed: No answer generated.")
            correct_count += 0  # Assume incorrect
            continue
        
        print(f"\nTrial {i}: {GREEN}{answer}{RESET}")
        
        judge_prompt = (
            f"You are judging the correctness of a test question.\n"
            f"Question: '{static_question}'\n"
            f"Ground Truth answer: '{correct_answer}'\n\n"
            f"Determine if the following answer is semantically equivalent to the ground truth answer (even if wording differs slightly). "
            f"Provide a brief reasoning, then end your response with exactly 'The answer is correct' if it matches the ground truth, or 'The answer is incorrect' if it doesn't.\n\n"
            f"Here is the answer to judge: '{answer}'\n"
        )
        judge_response = make_api_call(
            client, MODEL, [{"role": "user", "content": judge_prompt}]
        )
        full_judgment = judge_response.choices[0].message.content.strip()
        print(f"  Judgment {i} full response: {YELLOW}{full_judgment}{RESET}")
        
        # Extract Correct/Incorrect from the end, ignoring trailing punctuation
        cleaned_judgment = re.sub(r'[.!?]+$', '', full_judgment.lower().strip())
        match = re.search(r'(correct|incorrect)$', cleaned_judgment)
        judgment = match.group(1) if match else "incorrect"
        print(f"  Judgment {i} decision: {GREEN if judgment == 'correct' else RED}{judgment.upper()}{RESET}")
        
        if judgment == "correct":
            correct_count += 1
    return correct_count / len(answers)

def supervisor_improve_prompt(client, current_prompt, accuracy, prompt_history):
    """
    Supervisor Prompt Improvement: Generates an improved prompt based on current accuracy and history.
    The new prompt must be general and not contain any specific details about the question or answer.
    """
    history_str = ""
    for idx, (past_prompt, past_acc) in enumerate(prompt_history, 1):
        history_str += f"{idx}. Prompt: '{past_prompt}' \n\nEnd of Prompt {idx}. Accuracy: {past_acc * 100:.1f}%\n"

    improve_prompt = textwrap.dedent(f"""
        You are improving a prompt for an LLM to answer a question accurately.

        Current prompt: '{current_prompt}'

        History of past prompts and their accuracies:
{history_str}
        
        Suggest an improved version of the prompt to increase accuracy. Consider the weaknesses of LLMs when crafting an improved prompt. **Do not repeat any previous prompt.** Keep trying new things, beyond just small iterations; formulate hypotheses and run experiments to validate or invalidate them. Even if there is a pattern in your previous attempts, never mindlessly continue the pattern at the expense of trying a new hypothesis.

        The new prompt must be general and generic, without any specific hints, details, or references to the question, answer, or scenario. First reason about the results you've achieved so far, and then plan the next experiment you will perform. Then end your response with NEW PROMPT: <your new prompt>
    """).strip()

    print('imporve prompt', improve_prompt)
    response = make_api_call(
        client, MODEL, [{"role": "user", "content": improve_prompt}]
    )
    full_response = response.choices[0].message.content.strip()
    print(f"  Supervisor reasoning: {YELLOW}{full_response}{RESET}")
    
    # Extract the new prompt after "NEW PROMPT: " (case-sensitive)
    match = re.search(r'NEW PROMPT:\s*(.*)', full_response, re.DOTALL)
    new_prompt = match.group(1).strip() if match else full_response.strip()
    print(f"  Improved Prompt: {YELLOW}{new_prompt}{RESET}")
    return new_prompt

def main():
    """
    Main loop: Runs NUM_ITERATIONS iterations of QA trials, evaluation, and prompt improvement.
    """
    current_prompt = "Answer the following question."
    prompt_history = []
    print("Starting LLM system.\n")
    
    for iteration in range(NUM_ITERATIONS):
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
        print(f"Current Prompt: {current_prompt}\n")
        
        # Run trials
        answers = []
        for trial in range(NUM_TRIALS):
            try:
                answer = question_answerer(client, current_prompt, static_question)
                answers.append(answer)
            except Exception as e:
                print(f"Trial {trial + 1} failed: {e}")
                answers.append("")  # Append empty to maintain length
        
        # Evaluate
        try:
            accuracy = supervisor_evaluate(client, answers, correct_answer)
            print(f"\nAccuracy: {accuracy * 100:.1f}% ({int(accuracy * NUM_TRIALS)}/{NUM_TRIALS} correct)\n")
        except Exception as e:
            print(f"Evaluation failed: {e}")
            accuracy = 0.0
        
        # Improve prompt if not last iteration
        if iteration < NUM_ITERATIONS - 1:
            try:
                prompt_history.append((current_prompt, accuracy))
                current_prompt = supervisor_improve_prompt(client, current_prompt, accuracy, prompt_history)
                print("Prompt improved for next iteration.\n")
            except Exception as e:
                print(f"Prompt improvement failed: {e}\n")
                # Keep current prompt if improvement fails
        else:
            print("All iterations complete.")
    
    print("LLM system execution finished.")

if __name__ == "__main__":
    main()