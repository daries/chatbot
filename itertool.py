import itertools

# Definisikan hyperparameter yang diperbolehkan
layer1_neurons = list(range(16, 129,10))  # Minimal 16, maksimal 128 neuron di layer 1
layer2_neurons = list(range(8, 65,10))    # Minimal 8, maksimal 64 neuron di layer 2
layer3_neurons = [30]                  # Softmax layer dengan 30 neuron
epochs_range = list(range(10, 101,10))    #Minimal 5, maksimal 100 epoch

# Gabungkan semua kemungkinan kombinasi hyperparameter
hyperparameter_combinations = list(itertools.product(layer1_neurons, layer2_neurons, epochs_range))

# Cetak hasilnya
for i, (layer1, layer2, epoch) in enumerate(hyperparameter_combinations, 1):
    print(f"Kombinasi {i}: Layer 1 Neuron = {layer1}, Layer 2 Neuron = {layer2}, Layer 3 Neuron = {layer3_neurons[0]}, epoch = {epoch}")