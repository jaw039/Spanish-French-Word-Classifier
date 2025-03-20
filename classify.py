import numpy as np
from collections import Counter

def extract_word_dna(word):
    dna = []
    
    for letter in word:
        dna.append(f"letter_{letter}")
    
    for i in range(len(word) - 1):
        double = word[i:i+2]
        dna.append(f"duo_{double}")
    
    for tail_len in [1, 2, 3]:
        if len(word) >= tail_len:
            tail = word[-tail_len:]
            dna.append(f"tail_{tail}")
    
    for head_len in [1, 2, 3]:
        if len(word) >= head_len:
            head = word[:head_len]
            dna.append(f"head_{head}")
            
    size_bucket = min(len(word) // 2, 5)  
    dna.append(f"size_{size_bucket}")
    
    return dna

def build_language_fingerprints(words_list):
    fingerprint = Counter()
    
    spanish_giveaways = ["os", "ar", "er", "ir", "mente", "dad", "cion", "ll", "rr", "ia", "io", "ez", "ito", "ita"]
    french_giveaways = ["eu", "ou", "ai", "ei", "au", "eau", "oi", "ie", "tion", "eux", "aux", "ez", "ais", "ment"]
    
    for word in words_list:
        for letter in word:
            fingerprint[f"letter_{letter}"] += 1

        for i in range(len(word) - 1):
            double = word[i:i+2]
            fingerprint[f"duo_{double}"] += 1
            
        for i in range(len(word) - 2):
            triple = word[i:i+3]
            fingerprint[f"trio_{triple}"] += 1

        for tail_len in [1, 2, 3]:
            if len(word) >= tail_len:
                tail = word[-tail_len:]
                fingerprint[f"tail_{tail}"] += 3  

        for head_len in [1, 2, 3]:
            if len(word) >= head_len:
                head = word[:head_len]
                fingerprint[f"head_{head}"] += 2  

        size_bucket = min(len(word) // 2, 5)  
        fingerprint[f"size_{size_bucket}"] += 1
        
        for pattern in spanish_giveaways:
            if pattern in word:
                fingerprint[f"es_{pattern}"] += 3
                
        for pattern in french_giveaways:
            if pattern in word:
                fingerprint[f"fr_{pattern}"] += 3
    
    return fingerprint

def classify(train_words, train_labels, test_words):
    espanol_words = [w.lower() for w, label in zip(train_words, train_labels) if label == "spanish"]
    francais_words = [w.lower() for w, label in zip(train_words, train_labels) if label == "french"]
    
    word_count = len(train_words)
    espanol_prob = len(espanol_words) / word_count
    francais_prob = len(francais_words) / word_count
    
    espanol_prints = build_language_fingerprints(espanol_words)
    francais_prints = build_language_fingerprints(francais_words)
    
    espanol_total = sum(espanol_prints.values())
    francais_total = sum(francais_prints.values())
    
    vocab = set(list(espanol_prints.keys()) + list(francais_prints.keys()))
    vocab_size = len(vocab)
    
    results = []
    
    for mystery_word in test_words:
        mystery_word = mystery_word.lower()
        
        espanol_score = np.log(espanol_prob)
        francais_score = np.log(francais_prob)
        
        word_dna = extract_word_dna(mystery_word)
        
        for feature in word_dna:
            if feature in espanol_prints:
                espanol_score += np.log((espanol_prints[feature] + 1) / (espanol_total + vocab_size))
            else:
                espanol_score += np.log(1 / (espanol_total + vocab_size))
            
            if feature in francais_prints:
                francais_score += np.log((francais_prints[feature] + 1) / (francais_total + vocab_size))
            else:
                francais_score += np.log(1 / (francais_total + vocab_size))
        
        francais_score += 0.06
        
        if espanol_score > francais_score:
            results.append("spanish")
        else:
            results.append("french")
    
    return results