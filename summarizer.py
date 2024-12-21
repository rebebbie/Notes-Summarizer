import spacy
import pytextrank

# Create spaCy pipeline and add textrank to it

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")

try:
    with open("fulltext.txt", "r", encoding="utf-8") as file:
        example_text = file.read()
except FileNotFoundError:
    print("Error: The file 'fulltext.txt' was not found.")
    example_text = ""
    
if example_text:
    # Ask the user for the desired complexity level
    complexity = input("Choose the complexity level (short, medium, long): ").strip().lower()

    # Map complexity level to the number of sentences
    complexity_map = {
        "short": 2,
        "medium": 5,
        "long": 10
    }
    num_sentences = complexity_map.get(complexity, 5)  # Default to medium if input is invalid

    doc = nlp(example_text)

    # Generate summary
    print("\nSummary:")
    for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=num_sentences):
        print(sent)
else:
    print("The file is empty or could not be read.")


      


