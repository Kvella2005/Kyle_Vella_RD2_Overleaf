import time
import pandas as pd
from transformers import pipeline

# 1. Initialize the Zero-Shot Classifier
# We use BART-MNLI as specified in the research design
print("Loading model... this may take a moment.")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 2. Define Big Five Questions
# These are designed to elicit the 'candidate stories' mentioned in Task 1
questions = {
    "Openness": "Describe a time you used your imagination or a creative approach to solve a problem.",
    "Conscientiousness": "Describe a time you had to manage multiple deadlines. How did you stay organized?",
    "Extraversion": "Do you prefer collaborating with a large team or working independently? Why?",
    "Agreeableness": "Tell me about a time you had to deal with a difficult colleague. How did you handle it?",
    "Neuroticism": "Describe a high-pressure situation you faced recently and how you coped with it."
}

# 3. Collect User Input and Score
results = []
print("\n--- AI-Powered Personality Interview Prototype ---\n")

for trait, question in questions.items():
    print(f"Question ({trait}): {question}")
    user_response = input("Your Answer: ")
    
    # Start timer to measure efficiency
    start_time = time.time()
    
    # Perform Zero-Shot Classification
    # The model evaluates how well the response entails the personality trait label
    prediction = classifier(user_response, candidate_labels=[trait])
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Store numerical data for Pearson Correlation analysis
    results.append({
        "Trait": trait,
        "Response": user_response,
        "AI_Score": round(prediction['scores'][0], 4),
        "Time_Taken_Sec": round(processing_time, 2)
    })
    print(f"Scoring Complete. Score: {round(prediction['scores'][0], 4)}\n")

# 4. Export Data to CSV
# This supports the 'Quantitative Mono-method'
df = pd.DataFrame(results)
df.to_csv("interview_results.csv", index=False)

print("--- Interview Finished ---")
print("Results saved to 'interview_results.csv' for your correlation analysis.")