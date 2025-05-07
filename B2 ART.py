

import numpy as np

def similarity(input_pattern, weight):
    # Logical AND similarity
    overlap = np.sum(np.logical_and(input_pattern, weight))
    norm = np.sum(input_pattern)
    return overlap / (norm if norm != 0 else 1)

def update_weights(weight, input_pattern):
    return np.minimum(weight, input_pattern)

def process_input(input_pattern, categories, vigilance):
    input_pattern = np.array(input_pattern)
    best_similarity = -1
    best_category = -1

    # Compare similarities with exixting categories best similarity is choosen 
    for i, weight in enumerate(categories):
        sim = similarity(input_pattern, weight)
        if sim > best_similarity:
            best_similarity = sim
            best_category = i

    # If best similarity >= vigilance then assign it to best category, update weight and return category
    if best_category != -1 and best_similarity >= vigilance:
        print(f"Updating category {best_category} (similarity: {best_similarity:.2f})")
        categories[best_category] = update_weights(categories[best_category], input_pattern)
        return best_category
    # If best similarity >= vigilance then add ne category
    else:
        print("Creating a new category.")
        categories.append(input_pattern.copy())
        return len(categories) - 1

def train_art1(input_patterns, vigilance=0.8):
    categories = []
    #loop wil iterate till all patterns
    for idx, pattern in enumerate(input_patterns):
        print(f"\nInput Pattern {idx}: {pattern}")
        cat = process_input(pattern, categories, vigilance)
        print(f"Assigned to category: {cat}")
        print("Current Category Weights:")
        for j, weight in enumerate(categories):
            print(f"  Category {j}: {weight}")
    return categories

# Example input patterns
if __name__ == "__main__":
    input_data = [
        [1, 0, 1, 0, 1],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1]
    ]
    trained_categories = train_art1(input_data, vigilance=0.8)


# Adaptive Resonance Theory  by Stephen Grossberg and Gail Carpenter for unsupervised learning of binary input patterns (0s and 1s).

# Input pattern → F1
# Compute similarity with F2 nodes
# Choose best-matching F2 node
# Compare match with vigilance (ρ)
# If matched:
# Update weights (Hebbian)
# If not:
# Reset → try another node or create new category



