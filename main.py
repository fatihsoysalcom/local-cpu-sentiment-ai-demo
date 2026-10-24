import sys

try:
    from transformers import pipeline
except ImportError:
    print("This example requires the 'transformers' library.")
    print("Please install it using: pip install transformers torch")
    sys.exit(1)

def run_local_sentiment_analysis():
    """
    Demonstrates running a small, open-weight AI model locally on the CPU.
    This aligns with the EcoSloth AI philosophy of reusing older hardware
    for sustainable and private AI applications.
    """
    print("Initializing a local sentiment analysis model...")
    print("This model will run entirely on your CPU, demonstrating local AI inference.")

    # Initialize a sentiment analysis pipeline.
    # We explicitly set device='cpu' to ensure it runs on the CPU,
    # making it suitable for older or less powerful hardware.
    # The model 'distilbert-base-uncased-finetuned-sst-2-english' is a relatively
    # small, open-weight model suitable for local deployment.
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device="cpu" # Forces execution on CPU, ideal for older hardware
    )

    print("\nModel loaded successfully. Performing sentiment analysis locally...")

    texts = [
        "This is an excellent example of local AI!",
        "I am quite disappointed with the performance.",
        "The weather today is neutral.",
        "EcoSloth AI gives new life to old devices.", # Relates to article theme
        "This project is a fantastic idea for sustainability."
    ]

    for text in texts:
        # The inference happens entirely on your local machine,
        # without sending data to cloud services, enhancing privacy.
        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']
        print(f"Text: '{text}'")
        print(f"  Sentiment: {label} (Score: {score:.4f})")
        print("-" * 30)

if __name__ == "__main__":
    run_local_sentiment_analysis()
