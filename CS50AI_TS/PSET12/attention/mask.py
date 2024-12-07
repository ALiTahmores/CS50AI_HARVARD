import sys
import os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Set TensorFlow logging level to avoid unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants and model configuration
MODEL = "bert-base-uncased"  # Pre-trained BERT model
K = 5  # Number of predictions to generate
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)  # Font for visualizations
GRID_SIZE = 40  # Grid size for attention diagram
PIXELS_PER_WORD = 200  # Pixels per word in visualization


def main():
    """
    Main function to process the input text and generate predictions using BERT.
    It also visualizes the attention patterns.
    """
    try:
        text = input("Enter a text with a mask token (e.g., 'The capital of France is [MASK].'): ")
        if not text:
            raise ValueError("Input cannot be empty.")

        # Tokenize input
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        inputs = tokenizer(text, return_tensors="tf")
        mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
        if mask_token_index is None:
            raise ValueError(f"Input must include the mask token {tokenizer.mask_token}.")

        # Load the model and generate predictions
        model = TFBertForMaskedLM.from_pretrained(MODEL)
        result = model(**inputs, output_attentions=True)

        # Display predictions
        mask_token_logits = result.logits[0, mask_token_index]
        top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
        print("\nTop predictions for the masked word:")
        for token in top_tokens:
            print(f"Prediction: {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")

        # Visualize attention patterns
        visualize_attentions(inputs.tokens(), result.attentions)

    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_mask_token_index(mask_token_id, inputs):
    """
    Returns the index of the token with the specified `mask_token_id` in the input, or
    `None` if the mask token is not present.
    """
    for i, token in enumerate(inputs.input_ids[0]):
        if token == mask_token_id:
            return i
    return None


def get_color_for_attention_score(attention_score):
    """
    Converts attention score to a grayscale color value. Higher scores are lighter.
    """
    attention_score = attention_score.numpy()
    return (round(attention_score * 255), round(attention_score * 255), round(attention_score * 255))


def visualize_attentions(tokens, attentions):
    """
    Generates graphical visualizations of attention scores for each token.
    For each attention layer and head, a diagram is produced and saved.
    """
    for i, layer in enumerate(attentions):
        for k in range(len(layer[0])):
            layer_number = i + 1
            head_number = k + 1
            generate_diagram(layer_number, head_number, tokens, attentions[i][0][k])


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generates an attention heatmap diagram for a single attention head in a given layer.
    The diagram displays tokens on both axes, with color intensity representing attention scores.
    """
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw tokens on the grid
    for i, token in enumerate(tokens):
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw attention scores in grid
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save the diagram with a descriptive filename
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")
    print(f"Attention heatmap saved as Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
