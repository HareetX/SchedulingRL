def count_network_layers(network_config_path):
    layer_count = 0
    layers_section = False

    with open(network_config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "[LAYERS]":
                layers_section = True
                continue
            if layers_section and line and not line.startswith("#"):
                layer_count += 1

    return layer_count

if __name__ == "__main__":
    print(count_network_layers("configs/networks/test_network.cfg"))
