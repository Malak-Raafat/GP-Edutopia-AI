from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load API Key from environment variable
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set! Please check your environment variables.")

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.2-1b-preview",
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
U-Net Model
U-Net is a convolutional neural network architecture specifically designed for biomedical image 
segmentation. It has a symmetrical encoder-decoder structure, where the encoder extracts 
features, and the decoder reconstructs the image with segmentation masks. Skip connections link 
corresponding layers in the encoder and decoder to preserve spatial information. U-Net is highly 
efficient and performs well on small datasets, making it a popular choice in medical imaging tasks.
Architecture
Define U-Net Blocks:
• Implemented a convolutional block (conv_block) that includes two convolutional layers 
with ReLU activation, kernel initialization, and dropout for regularization.
• Created an upsampling block (upsample_block) using transposed convolution for 
upsampling and concatenation of features from previous layers.
Contracting Path:
• Used sequential convolutional blocks (conv_block) and max-pooling layers to reduce 
spatial dimensions while increasing the number of feature channels:
• Encoder: Extracts and compresses features from the input (downsampling).
Expanding Path:
• Applied upsampling blocks to reconstruct spatial dimensions and combine features from 
the contracting path:
• Decoder: Reconstructs the spatial dimensions and combines extracted features 
• These stages are connected by the bottleneck layer (c5), which acts as the transition point 
between the encoder and decoder.
Output Layer:
• Added a final convolutional layer with 1 filter and sigmoid activation to produce a 
probability map for binary segmentation.
• Model Training:
• Defined callbacks for early stopping and saving the best model:
o EarlyStopping monitored validation loss with a patience of 5 epochs.
o ModelCheckpoint saved the best model during training.
Model Saving:
• Saved the trained model in HDF5 format (model.h5).
SAM Model
SAM is based on a foundation of transformer models, leveraging the power of attention 
mechanisms to learn spatial relationships within images for precise segmentation. SAM uses a 
vision transformer (ViT) as its backbone. Vision transformers have self-attention mechanisms that 
allow the model to capture long-range dependencies between pixels.
Architecture:
The main parts of the SAM architecture include:
• Backbone (Vision Transformer - ViT): This is the core architecture of SAM, where image 
features are extracted.
• Prompt Encoder: This component processes the different types of input prompts (points, 
boxes, and masks) to guide the segmentation.
• Segmentation Decoder: This part decodes the model’s predictions into final segmentation 
masks.
Dice Loss Advantages:
• Handling Imbalanced Data: Dice Loss is particularly useful when the dataset is 
imbalanced.
Second: Plant Disease Recognition
Siamese Architecture: A neural network designed to determine the similarity or dissimilarity 
between two inputs.
Twin Networks: Consists of two identical sub-networks that share the same weights and 
parameters.
Shared Weights: Both sub-networks learn the same features from the input data, ensuring 
consistent comparisons.
Distance Metric: Outputs (feature vectors) from the sub-networks are compared using a 
distance metric like Euclidean distance or cosine similarity.
Training: Network is trained with pairs of images labeled as similar or dissimilar, adjusting 
parameters to bring similar images closer and dissimilar ones farther apart.
Application: Commonly used in tasks such as plant recognition or image matching where 
pairwise comparisons are necessary
Advantages of One-shot Learning in Plant Recognition:
• Reduced Data Requirements: Recognizes plant species with just one image per species, 
reducing the need for large labeled datasets.
• Generalization: Effectively generalizes to new, unseen plant species, especially with 
models like Siamese or Prototypical Networks.
AlexNet
AlexNet is a pioneering deep learning model that popularized convolutional neural networks in the 
2012 ImageNet competition. It uses five convolutional layers, followed by three fully connected 
layers, and employs techniques like ReLU activation, dropout, and data augmentation. AlexNet 
significantly reduced error rates at the time and laid the foundation for modern deep learning in 
computer vision.
Architecture:
Input:
• Accepts images of size 224x224x3 (RGB).
Feature Extraction (Convolutional and Pooling Layers):
• 5 convolutional layers: filters with ReLU activation.Followed by MaxPooling
Flatten and Dense Layers:
• Flatten: Converts the extracted features into a 1D vector.
• Dense Layer 1 & Dense Layer 2: 4096 units, with ReLU activation.Followed by Dropout (rate 
0.5) to reduce overfitting.
Output Layer: A dense layer with num_classes units and softmax activation
Optimization:
• Uses Adam optimizer, categorical cross-entropy loss, and accuracy as a performance 
metric.
Output: Produces class probabilities for classification tasks 
Worst Model: Context Matters
o Why: While AlexNet was groundbreaking in 2012, its architecture is now 
considered outdated compared to more efficient and deeper models like VGG and 
MobileNet. It has fewer layers, lower accuracy, and lacks optimizations like 
depthwise separable convolutions.
o Drawback: Inefficiencies and limitations make it less competitive in scenarios 
where computational resources and accuracy are critical

"""

# Change this value before each run
user_request = "Generate 10 y/n questions with answers and 100 t/f without answers and 60 wh questions without answers and 4 mcq with answers"

generate_questions(paragraph_text, user_request)