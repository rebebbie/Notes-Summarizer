from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging
import torch

# Enable logging to verify cache paths
#logging.set_verbosity_info()

# Pick model
model_name = "google/pegasus-xsum"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#pegasus_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
#pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

example_text = """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the structured part."""

# Create tokens
tokens = pegasus_tokenizer(example_text, truncation=True, padding="longest", return_tensors="pt")

# Summarize text
encoded_summary = pegasus_model.generate(**tokens)

# Decode summarized text
decoded_summary = pegasus_tokenizer.decode(
      encoded_summary[0],
      skip_special_tokens=True
)

# Define summarization pipeline 
summarizer = pipeline(
    "summarization", 
    model=model_name, 
    tokenizer=pegasus_tokenizer, 
    framework="pt"
)

# Create summary 
summary = summarizer(example_text, min_length=30, max_length=150)

print(summary[0]["summary_text"])
print("reached")