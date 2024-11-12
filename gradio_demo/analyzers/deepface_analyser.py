from deepface import DeepFace

class DeepFaceAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.gender = None
        self.age = None
        self.race = None 
        self.emotion = None  

    def analyze(self):
        results = DeepFace.analyze(img_path=self.image_path, actions=['gender', 'age', 'race', 'emotion'])
        
        # Extract specific details and store them as instance attributes
        self.gender = results[0]['gender']
        self.age = results[0]['age']
        self.race = results[0]['dominant_race'] 
        self.emotion = results[0]['dominant_emotion'] 
        
    def get_results(self):
        # Return the attributes as a dictionary for easy access
        return {
            "gender": self.gender,
            "age": self.age,
            "race": self.race,
            "emotion": self.emotion
        }
    
    def format_text_with_results(self, text):
        # Format the input text with available results
        return text.format(
            gender=self.gender or "unknown",
            age=self.age or "unknown",
            race=self.race or "unknown",
            emotion=self.emotion or "unknown"
        )
