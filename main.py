import torch
from utils.metrics import AverageMeter, calculate_overlap_metrics, calculate_f1_score
from model import Model
import gc
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.data_loader import Covid
import json
import argparse
import logging
import os

############# Define Parser #######################
parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--patience", type=int, default=300)
parser.add_argument("--best_acc", type=int, default=0)
parser.add_argument("--save_every", type=int, default=10)
parser.add_argument("--alpha", type=int, default=1)

args = parser.parse_args()

learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size
patience = args.patience
best_acc = args.best_acc
save_every = args.save_every
alpha = args.alpha

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device selected: ", device)

# Load config from JSON file
with open("model_config.json", "r") as f:
    model_config = json.load(f)

custom_model = Model(**model_config)
model = custom_model.get_model()

torch.cuda.empty_cache()
gc.collect()
print(model_config)

################### Define Logging #####################
log_file = f"logs/training/alpha_{alpha}/{custom_model.encoder_name}_{custom_model.decoder_name}"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to the console
        logging.FileHandler(log_file),  # Log to a file
    ],
)

# Log the model information
logging.info(
    f"Model Information: \n Encoder Name: {custom_model.encoder_name}, Decoder Name: {custom_model.decoder_name}"
)

# Log the information
logging.info(
    f"Training Information: \n Learning Rate: {learning_rate}, Num Epochs: {num_epochs}, Batch Size: {batch_size}, Patience: {patience}, Best Accuracy: {best_acc}, Save Every: {save_every}, Alpha: {alpha}"
)


############ Define Metric ###############
pixel_acc_infected_meter = AverageMeter()
dice_infected_meter = AverageMeter()
iou_infected_meter = AverageMeter()
precision_infected_meter = AverageMeter()
recall_infected_meter = AverageMeter()

pixel_acc_lungs_meter = AverageMeter()
dice_lungs_meter = AverageMeter()
iou_lungs_meter = AverageMeter()
precision_lungs_meter = AverageMeter()
recall_lungs_meter = AverageMeter()

precision_classification_meter = AverageMeter()
recall_classification_meter = AverageMeter()
f1_score_classification_meter = AverageMeter()


# Set up data loaders
train_data = Covid("data/Infection Segmentation Data/Infection Segmentation Data")
val_data = Covid(
    "data/Infection Segmentation Data/Infection Segmentation Data",
    mode="val",
)

train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)
# Set up model
model = model.to(device)

# Set up loss function
classification_loss_fn = nn.CrossEntropyLoss()
# segmentation_loss_fn = torchvision.ops.sigmoid_focal_loss
segmentation_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")


# Set up training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    print(f">>> Training epoch {epoch}")
    progress = tqdm(train_loader, total=int(len(train_loader)))
    for batch_idx, (
        inputs,
        labels_classification,
        labels_segmentation_lungs,
        labels_segmentation_infected,
    ) in enumerate(progress):
        # Training
        progress.refresh()
        # To device
        inputs = inputs.to(device)
        labels_classification = labels_classification.to(device)
        labels_segmentation_infected = labels_segmentation_infected.to(device)
        labels_segmentation_lungs = labels_segmentation_lungs.to(device)

        # Zero the parameter gradient
        optimizer.zero_grad()
        # Forward pasé
        (
            outputs_classification,
            outputs_segmentation_lungs,
            outputs_segmentation_infected,
        ) = model(inputs)

        outputs_classification = outputs_classification.type(torch.float32)
        outputs_segmentation_infected = outputs_segmentation_infected.type(
            torch.float32
        )
        outputs_segmentation_lungs = outputs_segmentation_lungs.type(torch.float32)

        labels_classification = labels_classification.type(torch.float32)
        labels_segmentation_infected = labels_segmentation_infected.type(torch.float32)
        labels_segmentation_lungs = labels_segmentation_lungs.type(torch.float32)
        #         print(outputs_classification ,labels_classification)

        loss_classification = classification_loss_fn(
            outputs_classification, labels_classification
        )
        loss_segmentation_infected = segmentation_loss_fn(
            outputs_segmentation_infected, labels_segmentation_infected
        )
        loss_segmentation_lungs = segmentation_loss_fn(
            outputs_segmentation_lungs, labels_segmentation_lungs
        )
        #         loss = (1/3 * loss_classification) + (1/3 * loss_segmentation_infected) + (1/3 * loss_segmentation_lungs)

        # loss = ((a * loss_classification) + (b * loss_segmentation_infected) + (c * loss_segmentation_lungs))/(10)
        loss = (loss_segmentation_infected) + (
            alpha * (loss_classification + loss_segmentation_lungs)
        )

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=loss.item())

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (
            inputs,
            labels_classification,
            labels_segmentation_lungs,
            labels_segmentation_infected,
        ) in enumerate(val_loader):
            # To device
            inputs = inputs.to(device)
            labels_classification = labels_classification.to(device)
            labels_segmentation_infected = labels_segmentation_infected.to(device)
            labels_segmentation_lungs = labels_segmentation_lungs.to(device)

            (
                outputs_classification,
                outputs_segmentation_lungs,
                outputs_segmentation_infected,
            ) = model(inputs)

            outputs_classification = outputs_classification.type(torch.float32)
            outputs_segmentation_infected = outputs_segmentation_infected.type(
                torch.float32
            )
            outputs_segmentation_lungs = outputs_segmentation_lungs.type(torch.float32)

            labels_classification = labels_classification.type(torch.float32)
            labels_segmentation_infected = labels_segmentation_infected.type(
                torch.float32
            )
            labels_segmentation_lungs = labels_segmentation_lungs.type(torch.float32)

            #             print(outputs_classification ,labels_classification)

            loss_classification = classification_loss_fn(
                outputs_classification, labels_classification
            )
            loss_segmentation_infected = segmentation_loss_fn(
                outputs_segmentation_infected, labels_segmentation_infected
            )
            loss_segmentation_lungs = segmentation_loss_fn(
                outputs_segmentation_lungs, labels_segmentation_lungs
            )
            #         loss = (1/3 * loss_classification) + (1/3 * loss_segmentation_infected) + (1/3 * loss_segmentation_lungs)
            loss = (
                (1 / 3 * loss_classification)
                + (1 / 3 * loss_segmentation_infected)
                + (1 / 3 * loss_segmentation_lungs)
            )
            val_loss += loss.item() * inputs.size(0)

            outputs_classification = (
                outputs_classification.argmax(1).detach().cpu().numpy()
            )
            outputs_segmentation_infected = outputs_segmentation_infected.argmax(1)
            outputs_segmentation_lungs = outputs_segmentation_lungs.argmax(1)

            labels_classification = (
                labels_classification.argmax(1).detach().cpu().numpy()
            )
            labels_segmentation_infected = labels_segmentation_infected.argmax(1)
            labels_segmentation_lungs = labels_segmentation_lungs.argmax(1)

            (
                pixel_acc_infected,
                dice_infected,
                iou_infected,
                precision_infected,
                recall_infected,
            ) = calculate_overlap_metrics(
                labels_segmentation_infected, outputs_segmentation_infected, eps=1e-5
            )
            (
                pixel_acc_lungs,
                dice_lungs,
                iou_lungs,
                precision_lungs,
                recall_lungs,
            ) = calculate_overlap_metrics(
                labels_segmentation_lungs, outputs_segmentation_lungs, eps=1e-5
            )
            precision_classification = precision_score(
                labels_classification, outputs_classification, average="macro"
            )
            recall_classification = recall_score(
                labels_classification, outputs_classification, average="macro"
            )
            f1_score_classification = f1_score(
                labels_classification, outputs_classification, average="macro"
            )

            pixel_acc_infected_meter.update(pixel_acc_infected, inputs.shape[0])
            dice_infected_meter.update(dice_infected, inputs.shape[0])
            iou_infected_meter.update(iou_infected, inputs.shape[0])
            precision_infected_meter.update(precision_infected, inputs.shape[0])
            recall_infected_meter.update(recall_infected, inputs.shape[0])

            pixel_acc_lungs_meter.update(pixel_acc_lungs, inputs.shape[0])
            dice_lungs_meter.update(dice_lungs, inputs.shape[0])
            iou_lungs_meter.update(iou_lungs, inputs.shape[0])
            precision_lungs_meter.update(precision_lungs, inputs.shape[0])
            recall_lungs_meter.update(recall_lungs, inputs.shape[0])

            precision_classification_meter.update(
                precision_classification, inputs.shape[0]
            )
            recall_classification_meter.update(recall_classification, inputs.shape[0])
            f1_score_classification_meter.update(
                f1_score_classification, inputs.shape[0]
            )
    #             f1_score(y_true, y_pred, average='macro')
    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)

    f1_score_infected_meter = calculate_f1_score(
        precision_infected_meter.avg, recall_infected_meter.avg
    )
    f1_score_lungs_meter = calculate_f1_score(precision_lungs_meter.avg, recall_lungs_meter.avg)

    logging.info(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} \n \
    pixel_acc_infected: {pixel_acc_infected_meter.avg :.4f}, dice_infected: {dice_infected_meter.avg :.4f},iou_infected: {iou_infected_meter.avg :.4f}, precision_infected: {precision_infected_meter.avg :.4f}, recall_infected: {recall_infected_meter.avg :.4f}, f1_score_infected: {f1_score_infected_meter :.4f} \n \
    pixel_acc_lungs: {pixel_acc_lungs_meter.avg :.4f}, dice_lungs: {dice_lungs_meter.avg :.4f},iou_lungs: {iou_lungs_meter.avg :.4f}, precision_lungs: {precision_lungs_meter.avg :.4f}, recall_lungs: {recall_lungs_meter.avg :.4f}, f1_score_lungs: {f1_score_lungs_meter :.4f} \n\
     precision_classification: {precision_classification_meter.avg :.4f}, recall_classification: {recall_classification_meter.avg :.4f},f1_score_classification: {f1_score_classification_meter.avg :.4f} \n"
    )

    saving_folder = f"checkpoints/alpha_{alpha}/{custom_model.encoder_name}_{custom_model.decoder_name}"
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # save models
    if iou_infected_meter.avg > best_acc:  # best base on ind
        logging.info(f"Best model found at epoch {epoch+1}, saving model")
        torch.save(
            model.state_dict(), f"{saving_folder}/sample_best.ckpt"
        )  # only save best to prevent output memory exceed error
        #         torch.save(model,'best.pth')
        best_acc = f1_score_classification_meter.avg
        stale = 0
    else:
        stale += 1
        if stale > patience:
            logging.info(f"No improvment {patience} consecutive epochs, early stopping")
            break
    if epoch % save_every == 0 or epoch == num_epochs - 1:
        logging.info(f"save model at epoch {epoch+1}, saving model")

        torch.save(model.state_dict(), f"{saving_folder}/epoch_{epoch}.ckpt")
