import streamlit as st
import numpy as np
import numpy.linalg as la

# Compute Cosine Similarity
def cosine_similarity(x,y):

    x_arr = np.array(x)
    y_arr = np.array(y)

    ############################
    ### WRITE YOUR CODE HERE ###
    ############################
    dot_product = np.dot(x_arr, y_arr)
    norm_x = la.norm(x_arr)
    norm_y = la.norm(y_arr)
    return dot_product / (norm_x * norm_y)


# Function to Load Glove Embeddings
def load_glove_embeddings(glove_path="Data/glove.6B.50d.txt"):
    """
    First step: Download the 50d Glove embeddings from here - https://www.kaggle.com/datasets/adityajn105/glove6b50d
    Second step: Format the glove embeddings into a dictionary that goes from a word to the 50d embedding.
    Third step: Store the 50d Glove embeddings in a pickle file of a dictionary.
    Now load that pickle file back in this function
    """

    embeddings_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    return embeddings_dict

# Get Averaged Glove Embedding of a sentence
def averaged_glove_embeddings(sentence, embeddings_dict):
    """
    Simple sentence embedding: Embedding of a sentence is the average of the word embeddings
    """
    words = sentence.split(" ")
    glove_embedding = np.zeros(50)
    count_words = 0

    ############################
    ### WRITE YOUR CODE HERE ###
    ############################

    words = sentence.lower().split(" ")
    glove_embedding = np.zeros(50)
    count_words = 0
    for word in words:
        if word in embeddings_dict:
            glove_embedding += embeddings_dict[word]
            count_words += 1
    if count_words > 0:
        glove_embedding /= count_words
    return glove_embedding


# Define your image paths here
category_images = {
    "flower": "Image/rose.jpg",
    "vehicle": "Image/tesla.png",
    "tree": "Image/pine.jpg",
    "mountain": "Image/mount_rainier.jpg",
    "building": "Image/paul_allen_center.jpg"
}

# Load GloVe embeddings
glove_embeddings = load_glove_embeddings()

# Define your Streamlit app
st.title("Semantic Image Search")

# Text input for user query
input_word = st.text_input("Enter a word to find the related image:", "")

if input_word:
    # Get the averaged embedding for the input
    input_embedding = averaged_glove_embeddings(input_word, glove_embeddings)
    
    # Compute similarity with each category
    cosine_sim = {}
    for category in category_images.keys():
        category_embedding = averaged_glove_embeddings(category, glove_embeddings)
        cosine_sim[category] = cosine_similarity(input_embedding, category_embedding)
    
    # Find the closest category
    closest_category = max(cosine_sim, key=cosine_sim.get)
    
    # Display the image corresponding to the closest category
    image_path = category_images[closest_category]
    st.image(image_path, caption=f"Image related to {closest_category}")