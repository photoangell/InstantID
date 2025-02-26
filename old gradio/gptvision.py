import base64
from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "images/nathan_sample1.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the image and return a description in the exact following format:\n\n"
                        "[age] years old [race] [gender] MMA fighter, [hair details] hair\n\n"
                        "Replace placeholders with actual details. Ensure age is a number, race is clear, "
                        "gender is male or female, and hair description includes length, texture, and color. "
                        "Do not include emotions, clothing, or background. Be concise and structured exactly as requested."
                    )
                    #"text": "Describe the persons gender, age with a number, race, face and hair type and color in the image, use words that would be used in a Stable Diffusion image prompt. Do not use filler words, do not describe the persons emotion, be concise and clear. respond in one sentance"
                    #"text":"how old is the person in the image in years? make your best guess. just give a number"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)