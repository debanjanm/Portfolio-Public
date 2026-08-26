import argparse

from pipeline.flow import MentalChatbot


def main():
    parser = argparse.ArgumentParser(description="Mental health RAG chatbot — single query")
    parser.add_argument("--query", required=True, help="user message to send")
    parser.add_argument("--user-id", default="cli-user", help="user identifier for chat history")
    parser.add_argument("--conversation-id", default="cli-session", help="conversation identifier for chat history")
    args = parser.parse_args()

    pipeline = MentalChatbot()
    response = pipeline.executing(args.query, args.user_id, args.conversation_id)
    print(response)


if __name__ == "__main__":
    main()
