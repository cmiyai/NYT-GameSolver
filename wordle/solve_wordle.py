from datetime import datetime, timedelta
import json
import os
import urllib.request
import numpy as np
import solver

now = datetime.now()
yester = now - timedelta(days=5)
yesterday = yester.strftime("%Y-%m-%d")
today = now.strftime("%Y-%m-%d")
tmrw = now + timedelta(days=1)
tmrw_date = tmrw.strftime("%Y-%m-%d") 
# run the wordle one day ahead so we can scan ahead of frontend
url = f"https://www.nytimes.com/svc/wordle/v2/{tmrw_date}.json"

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

try:
    with urllib.request.urlopen(req) as response:
        wordle = json.loads(response.read().decode())
        print(f"Wordle #{wordle['id']} ({wordle['print_date']}): {wordle['solution'].upper()}")
except Exception as e:
    raise SystemExit(f"Failed to fetch Wordle: {e}")

wordle_solution = wordle['solution'].lower()
puzzle_date = wordle['print_date']

STATUS = {0: "absent", 1: "correct", 2: "present"}
THRESHOLD = 5 # number of words that trigger when to only guess from:tune to best for sampling down when few words left


def best_entropy_guess(candidates, words_left, pattern_matrix, word_pool):
    """Returns the word from `candidates` with the highest Shannon entropy
    (i.e. the guess expected to narrow down `words_left` the most)."""
    best_word, best_score = None, -1
    for word in candidates:
        probs = solver.pattern_probabilities(word, words_left, pattern_matrix, word_pool)
        score = solver.ShannonEntropy(probs)
        if score > best_score:
            best_word, best_score = word, score
    return best_word, best_score


print("Loading data arrays...")
pattern_matrix = np.load("wordle_pattern_matrix.npy")
word_pool = np.loadtxt("words/valid-wordle-words.txt", dtype=str)
valid_answers = np.loadtxt("words/valid-answers.txt", dtype=str)

initial_pool_size = len(valid_answers)
words_left = np.arange(initial_pool_size)
steps = []

print(f"\n--- Auto-Solving Wordle #{wordle['id']} ---")

for turn in range(1, 7):
    pool_before = len(words_left)

    if pool_before == 1:
        guess = valid_answers[words_left[0]]
    else:
        # guess only words from valid if less than threshold
        candidates = valid_answers[words_left] if pool_before <= THRESHOLD else word_pool
        guess, _ = best_entropy_guess(candidates, words_left, pattern_matrix, word_pool)

    entropy_bits = solver.ShannonEntropy(solver.pattern_probabilities(guess, words_left, pattern_matrix, word_pool))
    pattern = solver.get_pattern(guess, wordle_solution)
    result = [{"letter": g.upper(), "status": STATUS[p]} for g, p in zip(guess, pattern)]
    print(f"Turn {turn}: guessing '{guess}' ({entropy_bits:.4f} bits, {pool_before} words left) -> {pattern}")

    target_pattern_id = solver.pattern_to_id(pattern)
    guess_idx = np.where(word_pool == guess)[0][0]
    matching_columns = np.where(pattern_matrix[guess_idx, :] == target_pattern_id)[0]
    words_left = np.intersect1d(words_left, matching_columns)
    remaining = len(words_left)

    eliminated_pct = round((pool_before - remaining) / pool_before * 100, 1) if pool_before else 0.0
    top = solver.top_candidates(valid_answers[words_left], words_left, pattern_matrix, word_pool, min(5, remaining))
    if len(words_left) < 10:
        print(valid_answers[words_left])
        for i in words_left:
            entro_guess = valid_answers[i]
            bro = solver.ShannonEntropy(solver.pattern_probabilities(entro_guess, words_left, pattern_matrix, word_pool))
            print(f"{entro_guess}: {bro} bits")

    steps.append({
        "turn": turn,
        "guess": guess.upper(),
        "result": result,
        "entropyBits": round(float(entropy_bits), 2) + 0.0,
        "remainingWords": remaining,
        "eliminatedPercentage": eliminated_pct,
        "topCandidates": top,
    })

    solved = guess == wordle_solution
    if solved or remaining == 0:
        break
else:
    solved = False

if not solved:
    print(f"\nFailed to solve, not in solution dictionary. The word was: {wordle_solution.upper()}")
else:
    print(f"\nSolved in {len(steps)} guess{'es' if len(steps) != 1 else ''}: {wordle_solution.upper()}")

result_data = {
    "date": puzzle_date,
    "wordle": {
        "date": puzzle_date,
        "answer": wordle_solution.upper(),
        "totalTurns": len(steps),
        "solved": solved,
        "initialPoolSize": initial_pool_size,
        "algorithmName": "Information Entropy",
        "steps": steps,
    },
}

output_dir = os.path.join("..", "frontend", "public")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{puzzle_date}_result.json")
with open(output_path, "w") as f:
    json.dump(result_data, f, indent=2)
print(f"Saved results to {output_path}")
