from pipeline.flow import MentalChatbot

# query = "Hi! I am Debanjan"
# query = "I am feeling sleepy"
query = "Can you tell me My Name?"


pipeline = MentalChatbot()
user_id, conversation_id = "1234", "abcd"
response = pipeline.executing(query, user_id, conversation_id)
print(response)