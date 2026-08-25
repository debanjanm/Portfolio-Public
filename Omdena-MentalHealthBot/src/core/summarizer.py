from transformers import pipeline


def summary(thread):
    pipe = pipeline(
        "summarization",
        model="kabita-choudhary/finetuned-bart-for-conversation-summary",
    )
    summ = pipe(thread)
    return summ
