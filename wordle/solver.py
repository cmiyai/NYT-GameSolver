import sys
import numpy as np

def get_pattern(guess: str, target: str):
    """
    Compares a guess to target word and returns length 5 tuple
    Gray = 0, Green = 1, Yellow = 2
    """

    try:
        if len(guess) != 5 or len(target) != 5:
            raise ValueError("Both guess and target must be 5 characters")
    except ValueError as e:
        sys.exit(f"Error {e}")

    pattern = [0]*5
    target_letters = list(target)
    guess_letters = list(guess)

    for i in range(5):
        if guess_letters[i] == target_letters[i]:
            pattern[i] = 1 # match so set to green
            target_letters[i] = None # dont allow for double count in yellow
            guess_letters[i] = None
    
    for i in range(5):
        if guess_letters[i] != None:
            if guess_letters[i] in target_letters:
                # if theres 2 of a letter in a word and we guess both we set other to yellow for mismatch
                pattern[i] = 2 # letter in word set to yellow
                target_letters[target_letters.index(guess_letters[i])] = None
    
    return tuple(pattern)


def pattern_to_id(pattern):
    """Converts a pattern tuple like (0, 2, 1, 0, 0) into a single integer ID 0-242"""
    # note there are 243 possible patterns; 3^5
    return sum(val * (3 ** i) for i, val in enumerate(pattern))


def pattern_probabilities(guess, words_left, pattern_matrix, word_pool):
    """
    Probability of a guess being all gray
    If we pass the max into shannon entropy, we can minimize entropy and max out Info gain
    """
    guess_idx = np.where(word_pool == guess)[0][0]

    active_patterns = pattern_matrix[guess_idx, words_left]

    counts = np.bincount(active_patterns, minlength=243)

    return counts / len(words_left)

def ShannonEntropy(dist):
    # dist = np.array(dist)
    dist =  dist[dist > 0]
    randos = np.where(dist > 0, dist * np.log2(dist) , 0)
    return -np.sum(randos)
# print(ShannonEntropy([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))

def top_candidates(pool_words, words_left, pattern_matrix, word_pool, limit):
    """Ranks `pool_words` by entropy against `words_left`, returns the top `limit` as
    [{"word": WORD, "entropy": bits}, ...] sorted highest-entropy first."""
    scored = [(w, ShannonEntropy(pattern_probabilities(w, words_left, pattern_matrix, word_pool))) for w in pool_words]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"word": str(w).upper(), "entropy": round(float(s), 2) + 0.0} for w, s in scored[:limit]]



if __name__ == "__main__":
    print("Loading data arrays...")
    pattern_matrix = np.load("wordle_pattern_matrix.npy")
    word_pool = np.loadtxt("words/valid-wordle-words.txt", dtype=str)
    valid_answers = np.loadtxt("words/valid-answers.txt", dtype=str)

    # start with all columns active
    words_left = np.arange(len(valid_answers))
    
    print("\n--- Live Wordle Solver Ready ---")
    print("For colors, enter 5 numbers: 0=Gray, 1=Green, 2=Yellow (e.g., 02100)")
    
    # for 6 turns
    for turn in range(1, 7):
        print(f"\n--- TURN {turn} ---")
        print(f"Remaining possible secret words: {len(words_left)}")
        
        if len(words_left) <= 10:
            print(f"Possible answers left: {valid_answers[words_left]}")
            if len(words_left) == 1:
                print("🎉 That's the winning word! Game over.")
                break
        
        # 1. Ask you what word you typed into Wordle
        guess = input("Enter the word you guessed (or press Enter to run grid search): ").strip().lower()
        
        # Run option for Entropy
        if not guess:
            print("\nCalculating best choices...")
            
            # --- 1. Rank the remaining winnable secret words ---
            secret_choices = []
            for idx in words_left:
                word = valid_answers[idx]
                probs = pattern_probabilities(word, words_left, pattern_matrix, word_pool)
                score = ShannonEntropy(probs)
                secret_choices.append((word, score))
            
            # Sort remaining secrets by highest entropy
            secret_choices.sort(key=lambda x: x[1], reverse=True)
            
            print("\nTop remaining POSSIBLE WORDS (Can win the game):")
            print("-" * 45)
            for word, score in secret_choices[:5]:
                print(f"🎯 {word:<8} | {score:.4f} bits")
                
            # --- 2. Rank strategic filler words from the entire pool ---
            # Only run this if there are more than 2 secrets left to save time
            if len(words_left) > 2:
                all_choices = []
                for word in word_pool:
                    probs = pattern_probabilities(word, words_left, pattern_matrix, word_pool)
                    score = ShannonEntropy(probs)
                    all_choices.append((word, score))
                all_choices.sort(key=lambda x: x[1], reverse=True)
                
                print("\nTop STRATEGIC WORDS (Best information gain):")
                print("-" * 45)
                for word, score in all_choices[:10]:
                    print(f"💡 {word:<8} | {score:.4f} bits")
            print("-" * 45)
            
            guess = input("\nEnter the word you actually guessed: ").strip().lower()

        # 2. Ask for the color combination the game gave you
        pattern_input = input("Enter the 5-digit color code from Wordle: ").strip()
        
        # Convert your string input (e.g. "02100") into a numeric tuple (0, 2, 1, 0, 0)
        pattern_tuple = tuple(int(char) for char in pattern_input)
        
        # Convert that color tuple to its exact 0-242 ID number
        target_pattern_id = pattern_to_id(pattern_tuple)
        
        # 3. FILTER STEP: Look at the matrix row for your guess
        guess_idx = np.where(word_pool == guess)[0][0]
        row_patterns = pattern_matrix[guess_idx, :]
        
        # Only keep the columns where the pattern matches the real game feedback
        matching_columns = np.where(row_patterns == target_pattern_id)[0]
        words_left = np.intersect1d(words_left, matching_columns)