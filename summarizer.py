import spacy
import pytextrank
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline

try:
    with open("fulltext.txt", "r", encoding="utf-8") as file:
        full_text = file.read()
except FileNotFoundError:
    print("Error: The file 'fulltext.txt' was not found.")
    full_text = ""
    
# Helper function that splits text into "chunks" of 512 tokens
def split_into_chunks(text, tokenizer, max_length=500):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    for i in range(0, len(input_ids), max_length):
        yield tokenizer.decode(input_ids[i:i + max_length], skip_special_tokens=True)
    
if full_text:
    # Gives user the option to choose between three different lengths for
    #  the shortened text file (3/6/9 sentences)
    complexity = input("Choose the complexity level (short, medium, long): ").strip().lower()
    
    complexity_map = {
    "short": (3, 100),    # Short: 3 sentences, 100 characters
    "medium": (6, 200),   # Medium: 6 sentences, 200 characters
    "long": (9, 300)      # Long: 9 sentences, 300 characters
    }

    # Validate user input and get the appropriate values
    if complexity in complexity_map:
        num_sentences, num_characters = complexity_map[complexity]
        print("num_sentences = " + str(num_sentences))
        print("num_characters = " + str(num_characters))
    else:
        print("Invalid input. Defaulting to medium.")
        num_sentences, num_characters = complexity_map["medium"]
    
    # ----- SHORTENING -----
    # For the shortened version (extracting the "best" sentences from the text), 
    #  the spaCy pipeline is created and textrank was added
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textrank")
    document = nlp(full_text)
    
    shortened_text = ""
    
    # generated shortened file (testing, outputted to console)
    for sentence in document._.textrank.summary(limit_phrases=2, limit_sentences=num_sentences):
        shortened_text = shortened_text + str(sentence) + " "
        
    # Storing the shortened summary in shortenedtext.txt
    try:
        with open("shortenedtext.txt", "w", encoding="utf-8") as output_file:
            output_file.write(shortened_text)
        print("\nShortening was successful! The summary is now stored in shortenedtext.txt")
    except IOError:
        print("Error: Unable to write to 'shortenedtext.txt'.")
        
    # ----- ABSTRACTION SUMMARY -----
    # For the "abstract" version (actually summarizes the information),
    #  I'm using a PEGASUS tokenizer
    
    # Loading in the pre-trained model pegasus-xsum 
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Split text into chunks
    chunks = list(split_into_chunks(full_text, tokenizer))

    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        tokens = tokenizer(chunk, truncation=True, padding="longest", return_tensors="pt")
        encoded_summary = model.generate(**tokens)
        decoded_summary = tokenizer.decode(
            encoded_summary[0],
            skip_special_tokens=True
        )
        summarizer = pipeline(
            "summarization", 
            model=model_name, 
            tokenizer=tokenizer, 
            framework="pt"
        )
        # Creating summary with modified size
        summary = summarizer(chunk, min_length=30, max_length=num_characters)
        summaries.append(summary)

    # Combining the summaries
    final_summary = " "
    for summary in summaries:
        final_summary = final_summary + summary[0]["summary_text"] + " "
    
    # Creating summary with modified size
    if final_summary:
        summarized_text = final_summary
        try:
            with open("summarizedtext.txt", "w", encoding="utf-8") as summarized_file:
                summarized_file.write(summarized_text)
            print("\nAbstraction was successful! The summary is now stored in summarizedtext.txt")
        except IOError:
            print("Error: Unable to write to 'summarizedtext.txt'.")
    else:
        print("Error: No summary was generated.")
    
else:
    print("The file is empty or could not be read.")



      


