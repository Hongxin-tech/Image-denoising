from utils import load_trained_model, plot_comparison_with_noisy, process_full_image

# Load the trained model
model_path = r'Pre-traned models/trained_lista_model_convolutional_sigma20_snr15.pth'
loaded_net = load_trained_model(model_path)

# Define paths and model
input_image_path = r'path/of/sample/image'
noisy_image_path = r'path/of/sample/noisy/image'

# Process the noisy full image
reconstructed_image = process_full_image(input_image_path, loaded_net)
plot_comparison_with_noisy(input_image_path, noisy_image_path, reconstructed_image)
