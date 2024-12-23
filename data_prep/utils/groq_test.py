import os

from groq import Groq

# client = Groq(
#     api_key=os.environ.get("GROQ_API_KEY"),
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],
#     model="llama3-8b-8192",
# )

# print(chat_completion.choices[0].message.content)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
system_prompt = '''
You are a helpful assistant highly skilled at summarizing a news article by extracting relevant information about the major league cricket tournament from that article. 

You will be given a news article. Your job is to extract relevant information about the major league cricket tournament from that article and summarizing it. 

Make sure to use only the information present in the news article to come up with your summary.
'''

chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
        },
        {
                "role": "user",
                "content": "The Major League Cricket tournament is a cricket league in the United States. It is a professional Twenty20 cricket league. The league was founded in 2018 and is scheduled to begin in 2022. The league will feature six teams and will be played in a round-robin format",
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
    )

response = chat_completion.choices[0].message.content
# Remove any introductory text from the response
response = response[response.find(':')+1:].strip()
print(response)
