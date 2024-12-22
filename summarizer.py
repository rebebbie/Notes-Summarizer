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
    
if full_text:
    # Gives user the option to choose between three different lengths for
    #  the shortened text file (3/6/9 sentences)
    complexity = input("Choose the complexity level (short, medium, long): ").strip().lower()
    
    complexity_map = {
        "short": (3,100),
        "medium": (6,200), #change to 10/25/40% of original length 
        "long": (9,300)
    }
    
    # If input is invalid, default to 6
    num_sentences = complexity_map.get(complexity[0], 6)  
    
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

    # Creating tokens
    tokens = tokenizer(full_text, truncation=True, padding="longest", return_tensors="pt")
    
    # Using abstraction to summarize the text (expressed as tokens)
    encoded_summary = model.generate(**tokens)

    # Decoding encoded_summary 
    decoded_summary = tokenizer.decode(
          encoded_summary[0],
          skip_special_tokens=True
    )
    
    # Defining the summarization pipeline to adjust size of output
    summarizer = pipeline(
        "summarization", 
        model=model_name, 
        tokenizer=tokenizer, 
        framework="pt"
    )
    
    # Creating summary with modified size
    num_characters = complexity_map.get(complexity[1], 200)  
    summary = summarizer(full_text, min_length=30, max_length=num_characters)
    if summary:
        summarized_text = summary[0]["summary_text"]
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



      


