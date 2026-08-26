import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "models/Trained-DistilBertForSequenceClassification_6h_768dim"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "models/Trained-DistilBertForSequenceClassification_6h_768dim"
)


# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def sentiment_class(summarized_text):
    """
    # 1 = non-depressed
    # 0 = depressed
    returns: example:- array([[0.00493283, 0.9950671 ]], dtype=float32)
    """
    inputs = tokenizer(
        summarized_text, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    outputs = model(**inputs)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()
    return predictions


def pattern_classification():
    raise NotImplementedError


def corelation_analysis():
    raise NotImplementedError
