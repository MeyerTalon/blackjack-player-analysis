from httpcore._synchronization import AsyncSemaphore
from ollama import AsyncClient, Client
import time
import asyncio
from typing import List


# Aggressive player, high-risk
aggressive_player_prompt = """
You are an aggressive blackjack player who takes bold risks to maximize winnings. 
You frequently hit even with moderately strong hands, double down whenever possible, and rarely stand before reaching 
18 or higher. You may occasionally split nontraditional pairs to press your advantage. Your goal is to maximize profit 
in the short term, even if it increases the risk of busting. Always respond with your next move only, in one word: 
"hit", "stand", "double", "split", or "surrender".
"""

# Conservative player, risk-averse
conservative_player_prompt = """
You are a conservative blackjack player who prioritizes minimizing losses and avoiding busts. 
You almost never take unnecessary risks. You stand on safe hands (16 or higher against dealer’s weak cards), rarely 
double down, and avoid splitting unless it clearly benefits you (like Aces or 8s). Your goal is to survive as long as 
possible, even at the cost of potential profit. Always respond with your next move only, in one word: 
"hit", "stand", "double", "split", or "surrender".
"""

# Basic strategy player, mathematically optimal
optimal_player_prompt = """
You are a blackjack player who always follows the official basic strategy chart for standard rules 
(4–8 decks, dealer hits on soft 17, double after split allowed). Use mathematically optimal decisions based on the 
player’s total and the dealer’s upcard. Do not take risks or follow intuition—only make the statistically correct move 
according to basic strategy. Always respond with your next move only, in one word: "hit", "stand", "double", or "split".
"""

# Absolute idiot player, chooses random action
random_player_prompt = """
You are a blackjack player who chooses actions completely at random, without regard to strategy or hand value. 
Each time you are asked to act, you must randomly select one of the valid moves: "hit", "stand", "double", "split", or 
"surrender". Do not attempt to evaluate the cards or the dealer’s upcard—your move should be arbitrary. 
Always respond with a single word only: "hit", "stand", "double", "split", or "surrender".
"""


def single_prompt_speed_test(prompt: str) -> float:
    """
    Runs a single Ollama inference with gpt-oss:20b and returns the elapsed time in seconds.

    Args:
        prompt: A text prompt to send to the model for generation.

    Returns:
        The total elapsed time in seconds from request start to completion.
    """
    client = Client()  # defaults to http://localhost:11434
    resp = client.generate(
        model='gpt-oss:20b',
        prompt=prompt
    )
    inference_duration = resp['total_duration'] / 1e9   # convert total duration from nanoseconds to seconds
    return inference_duration


async def run_inference(client: AsyncClient, prompt: str, sem: asyncio.Semaphore):
    """
    Send a single prompt to gpt-oss:20b and return the response.


    """
    async with sem:  # limit concurrency
        print(f'-> Starting inference with prompt {prompt[:30]}...')
        response = await client.generate(
            model="gpt-oss:20b",
            prompt=prompt
        )
    return prompt, response

async def multiple_prompts_speed_test(prompts: List[str], num_prompts: int = 3) -> float:
    """
    Run multiple prompts concurrently against a local Ollama model to measure total execution time.

    Args:
        prompts: A list of text prompts to send to the model.
        num_prompts: The maximum number of concurrent inference requests to run (default is 3).

    Returns:
        The total time in seconds taken to complete all prompt inferences.
    """
    start = time.time()

    # Limit how many requests run concurrently
    sem = asyncio.Semaphore(3)
    client = AsyncClient()

    tasks = [run_inference(client, p, sem) for p in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, output in results:
        print(f"\nPrompt: {prompt}\n{'-' * 60}\n{output['response']}\n")
    end = time.time()
    elapsed = end - start
    print(f"Time taken: {elapsed:.3f}")
    return elapsed

if __name__ == "__main__":
    simple_bj_prompt = 'You are a blackjack player playing like the dealer meaning you always hit until you are at 17 or higher. You have been dealt a hard 10, the dealer is dealt an 8 face up. Respond in one word, either hit or stand.'
    simple_bj_prompts = [simple_bj_prompt for _ in range(5)]
    realistic_bj_prompts =[
        aggressive_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15.",
        conservative_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15.",
        optimal_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15.",
        random_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15.",
        optimal_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15."
    ]


    single_prompt_inference_times = [single_prompt_speed_test(simple_bj_prompt) for _ in range(5)]
    # five_prompts_inference_time = asyncio.run(multiple_prompts_speed_test(realistic_bj_prompts, num_prompts=5))

    print(f'Single prompt inference speeds: {single_prompt_inference_times} seconds')
    print(f'Single prompt inference speed total: {sum(single_prompt_inference_times):.3f} seconds')
    # print(f'{len(simple_bj_prompts)} simple prompts ran in parallel: {five_prompts_inference_time:.3f} seconds')
    # print(f'{len(simple_bj_prompts)} realistic prompts ran in parallel: {five_prompts_inference_time:.3f} seconds')
    # print(f'Extrapolating the 5 player round to a game of 40 rounds: {(40 * five_prompts_inference_time) / 60:.3f} minutes')
