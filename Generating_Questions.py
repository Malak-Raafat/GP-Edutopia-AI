from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

# Load API Key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set! Please check your environment variables.")

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7
)


# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that generates structured questions based on a given paragraph. "
               "Strictly follow the user's request exactly as provided."),
    ("user", "Paragraph: {paragraph}\nUser Request: {user_request}")
])

# Create the chain
chain = prompt_template | llm

def generate_questions(paragraph: str, user_request: str):
    """Generate questions based on the exact user request."""
    result = chain.invoke({
        "paragraph": paragraph,
        "user_request": user_request
    })
    
    print(result.content)  # Display the generated questions

# Example usage (Modify 'user_request' as needed)
paragraph_text = """First: Plant Classification  
Data Preparation 
1. Setting Constants
image_size: Defines the target size to which images will be resized (224x224 pixels).
batch_size: The number of images to process at once during training. (64 batches)
2. Loading Data
This function loads images and their corresponding labels from the specified directory.
3. Normalizing Images
The images are normalized by dividing each pixel's value by 255.0, converting the pixel values from 
the range [0, 255] to [0, 1]. This step helps improve the convergence during model training.
4. Splitting the Data into Train and Validation Sets
• This splits the loaded data into training and validation sets, with 80% of the data used for 
training and 20% for validation 
7. Label Encoding
• The LabelEncoder is used to convert text labels into integer labels 
8. Class Weight Calculation
• The class weights are computed to handle class imbalance. The compute_class_weight 
function calculates weights inversely proportional to class frequencies.
10. Data Augmentation
We apply augmentation only in MobileNet, VGG, AlexNet, as we don't apply augmentation in 
ViT. The train_datagen applies several data augmentation techniques to the training images, 
including: Random rotation (rotation_range), width/height shifting, shearing, zooming, and flipping 
the images horizontally. val_datagen doesn't apply any augmentation
These augmentations help improve model generalization by providing more varied input data.
11. Data Generators
train_generator and validation_generator and testing_generator are instances of 
ImageDataGenerator that yield batches of images and labels during training and validation, 
respectively
Vision Transformer (ViT)
Architecture
• Divides input images into fixed-size non-overlapping patches (e.g., 16×1616 \times 
1616x16).
• Converts patches into 1D vector embeddings via a linear projection.
• Adds positional embeddings to preserve spatial relationships.
• Processes embeddings using Transformer encoder layers (self-attention + feedforward 
networks).
• Uses a learnable [CLS] token for classification.
• Final output is passed to a classification head for tasks like image classification.
Advantages
• Scalability: Performs better with larger datasets.
• Global Context: Captures global relationships across the image.
• Flexibility: Can adapt to multi-modal tasks beyond vision (e.g., vision + text).
• Reduced Inductive Bias: Learns more adaptively compared to CNNs.
• Improved Performance: Outperforms CNNs on benchmarks when pre-trained on large 
datasets.
• Parallelization: Faster training due to sequence-level parallel processing.
• Transfer Learning: Pre-trained ViTs generalize well to other tasks.
Challenges
• Data Requirements: Needs large-scale datasets for effective training.
• Computational Cost: High memory and computation demands due to quadratic self attention complexity.
• Overfitting: Prone to overfitting on smaller datasets.
• Interpretability: Harder to interpret learned features compared to CNNs.MobileNet
MobileNet is a lightweight deep learning model designed for mobile and embedded devices, 
prioritizing efficiency and speed. It uses depthwise separable convolutions to reduce the number of 
parameters and computations. This architecture is well-suited for tasks like image classification 
and object detection on resource-constrained devices. Despite its simplicity, it achieves 
competitive accuracy compared to larger models.
Architecture:
Input: Images of size 224x224x3 (RGB).
Base Model:
• MobileNet (pre-trained on ImageNet, without the top classification layers).
• Lightweight and efficient architecture, designed with depthwise separable convolutions for 
reduced computational complexity.
• Base model layers are frozen (not trainable).
Custom Layers:
• Global Average Pooling (GAP): Reduces the spatial dimensions of the feature maps to a 
single vector for each channel, summarizing the spatial information globally.
• Dense Layer: Fully connected layer with 1024 units and ReLU activation.
• Dropout Layer: Dropout with a rate of 0.5 to reduce overfitting.
• Output Layer: Dense layer with num_classes units and softmax activation for classification.
Optimization:
• Uses Adam optimizer, categorical cross-entropy loss, and accuracy as a performance 
metric.
Output: Class probabilities for the given number of output classes 
 Best for Resource-Constrained Devices: MobileNet
• Why: MobileNet is optimized for efficiency and speed, making it ideal for mobile and 
embedded devices. Despite its smaller size, it delivers competitive performance on tasks 
like image classification and object detection.VGG16
VGG is a deep convolutional neural network known for its simplicity and uniform architecture, 
consisting of sequential 3x3 convolutional layers followed by fully connected layers. It comes in 
variations like VGG-16 and VGG-19, named for the number of layers. VGG models are 
computationally expensive but deliver high accuracy in image classification. Their deep and 
uniform structure has influenced the design of many subsequent models.
Architecture:
Input:
• Images of size 224x224x3 (RGB).
Base Model:
• VGG16 (pre-trained on ImageNet, without the top classification layers).
• Contains 13 convolutional layers grouped into 5 blocks, each followed by max-pooling 
layers for feature extraction.
Custom Layers:
• Flatten: Converts feature maps from VGG16 into a 1D vector.
• Dense Layer 1: Fully connected layer with 4096 units and ReLU activation.
• Dropout Layer 1: Dropout with a rate of 0.5 to reduce overfitting.
• Dense Layer 2: Fully connected layer with 4096 units and ReLU activation.
• Dropout Layer 2: Another dropout with a rate of 0.5.
• Output Layer: Dense layer with num_classes units and softmax activation for classification.
Optimization:
• Uses Adam optimizer, categorical cross-entropy loss, and accuracy as a performance 
metric.
 Output: Class probabilities for the given number of output classes 
Best for High Accuracy on Large Datasets: VGG
• Why: VGG models, particularly VGG-16 and VGG-19, provide high accuracy due to their 
deeper architecture and consistent design. They are well-suited for applications requiring 
precise feature extraction.

"""

# Change this value before each run
user_request = "Generate 10 y/n questions with answers and 100 t/f without answers and 60 wh questions without answers and 4 mcq with answers"

generate_questions(paragraph_text, user_request)
