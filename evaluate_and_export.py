import csv
import torch


def evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev,num_epochs,train_batch_size,val_accuracy,fel_le_kerekit,model_neve,t_loss_min):
    model.eval()  # Váltás kiértékelési módba

    results = []
    reverse_label_map = {idx: label for label, idx in label_map.items()}  # Címkék visszafejtése

    with torch.no_grad():  # Gradiensek nem szükségesek kiértékelés során
        for test_images, test_ids in zip(test_image_tensors, test_image_ids):
            test_images = test_images.unsqueeze(0).to(dev)

            outputs = model(test_images)
            _, predicted = torch.max(outputs, 1)

            predicted_label = reverse_label_map[predicted.item()]
            if fel_le_kerekit==1 :
                predicted_label =  int(abs(float(predicted_label)))
            else :
                predicted_label = round(abs(predicted_label))  # Feltöltés előtt kötelező módosítás

            results.append([test_ids, predicted_label])


    #    with open(output_file, mode='w', newline='') as file:
    #        writer = csv.writer(file)
    #        writer.writerow(['Id', 'Expected'])
    #        writer.writerows(results)




    import datetime
    today = datetime.datetime.now().strftime("%Y_%m_%d")
    if fel_le_kerekit == 1:
        output_file = f'results/{today}_solution_epo{num_epochs}_bat{train_batch_size}_acc{val_accuracy:.2f}_le_{model_neve}_tMinLoss{t_loss_min}.csv'
    else:
        output_file = f'results/{today}_solution_epo{num_epochs}_bat{train_batch_size}_acc{val_accuracy:.2f}_fel_{model_neve}_tMinLoss{t_loss_min}.csv'


    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Expected'])
        writer.writerows(results)

    print(f"Eredmények kiírva: {output_file}")
