from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def main():
    model_name = "google/pegasus-xsum"
    pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
    pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)
    print("Model and tokenizer downloaded successfully!")

if __name__ == '__main__':
    main()
