from deepface import DeepFace

# Path to the image you want to analyze
image_path = "/workspace/img/input/passenger/batch1/henry.jpg"

# Analyze the image
results = DeepFace.analyze(img_path=image_path, actions=['gender', 'age', 'race', 'emotion'])

# Extract specific details
gender = results[0]['gender']
age = results[0]['age']
dominant_race = results[0]['dominant_race']
dominant_emotion = results[0]['dominant_emotion']

# Print results
print("Gender:", gender)
print("Age:", age)
print("Skin Color/Ethnicity:", dominant_race)
print("Dominant Emotion:", dominant_emotion)
