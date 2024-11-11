import csv
import torch


def evaluate_model(model, test_image_tensors, test_image_ids, label_map, output_file, dev):
    """
    Kiértékelés és eredmények mentése CSV fájlba.

    Parameters:
    - model: A betanított PyTorch modell
    - test_image_tensors: Teszt képek tensorai
    - test_image_ids: Teszt kép ID-k
    - label_map: Eredeti címkék és indexek közötti megfeleltetés
    - output_file: A kimeneti CSV fájl neve
    - dev: Eszköz (CPU vagy GPU)
    """
    model.eval()  # Váltás kiértékelési módba

    results = []
    reverse_label_map = {idx: label for label, idx in label_map.items()}  # Címkék visszafejtése

    with torch.no_grad():  # Gradiensek nem szükségesek kiértékelés során
        for test_images, test_ids in zip(test_image_tensors, test_image_ids):
            test_images = test_images.unsqueeze(0).to(dev)

            outputs = model(test_images)
            _, predicted = torch.max(outputs, 1)

            predicted_label = reverse_label_map[predicted.item()]
            predicted_label =  int(abs(float(predicted_label))) #---------------------------------------------------------------------------------
            #predicted_label = round(abs(predicted_label))  # Feltöltés előtt kötelező módosítás

            results.append([test_ids, predicted_label])

    # Eredmények mentése CSV fájlba
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Expected'])
        writer.writerows(results)

    print(f"Eredmények kiírva: {output_file}")
