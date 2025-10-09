from ollama import AsyncClient, Client
import time
import asyncio
from typing import List
from src.blackjack_simulator.player_prompts import (
    aggressive_player_prompt,
    conservative_player_prompt,
    optimal_player_prompt,
    random_player_prompt
)


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


async def run_inference(client: AsyncClient, prompt: str):
    """
    Send a single prompt to gpt-oss:20b and return the response.


    """
    async with sem:  # limit concurrency
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

    global sem

    # Limit how many requests run concurrently
    sem = asyncio.Semaphore(num_prompts)
    client = AsyncClient()

    tasks = [run_inference(client, p) for p in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, output in results:
        print(f"\nPrompt: {prompt}\n{'-' * 60}\n{output['response']}\n")
    end = time.time()
    elapsed = end - start
    print(f"Time taken: {elapsed:.3f}")
    return elapsed

if __name__ == "__main__":
    simple_bj_prompt = 'You are a blackjack player playing like the dealer meaning you always hit until you are at 17 or higher. You have been dealt a 10. Respond in one word, either hit or stand.'
    simple_bj_prompts = [simple_bj_prompt for _ in range(5)]
    realistic_bj_prompts =[
        aggressive_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15.",
        conservative_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15.",
        optimal_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15.",
        random_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15.",
        optimal_player_prompt + "The dealer is dealt a 9 face up, you are dealt a hard 15."
    ]


    # single_prompt_inference_time = single_prompt_speed_test(simple_bj_prompt)
    five_prompts_inference_time = asyncio.run(multiple_prompts_speed_test(realistic_bj_prompts, num_prompts=5))

    # print(f'Single prompt inference speed: {single_prompt_inference_time:.3f} seconds')
    # print(f'{len(simple_bj_prompts)} simple prompts ran in parallel: {five_prompts_inference_time:.3f} seconds')
    print(f'{len(simple_bj_prompts)} realistic prompts ran in parallel: {five_prompts_inference_time:.3f} seconds')
    print(f'Extrapolating the 5 player round to a game of 40 rounds: {(40 * five_prompts_inference_time) / 60:.3f} minutes')
