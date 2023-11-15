import torch
import torch
from utils.metrics import (
    AverageMeter,
    calculate_overlap_metrics,
    calculate_overlap_metrics_post,
    calculate_f1_score,
    export_to_excel,
    count_parameters
)
from model import Model
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.data_loader import Covid
import json
import argparse
import logging
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.post_processing import noise_remove, post_processing
import argparse
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device selected: ", device)


parser = argparse.ArgumentParser()

parser.add_argument("--weight_name", type=str, default="epoch_199.ckpt")
args = parser.parse_args()

weight_name = args.weight_name

# Load config from JSON file
with open("test_model_config.json", "r") as f:
    model_config = json.load(f)

custom_model = Model(**model_config)
model = custom_model.get_model()
model = model.to(device)

torch.cuda.empty_cache()
gc.collect()

weight_file = f"checkpoints/alpha_1/{custom_model.encoder_name}_{custom_model.decoder_name}/{weight_name}"

model.load_state_dict(torch.load(weight_file, map_location=device))
model.eval()


################### Define Logging #####################
log_file = (
    f"logs/testing/alpha_1/{custom_model.encoder_name}_{custom_model.decoder_name}"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to the console
        logging.FileHandler(log_file),  # Log to a file
    ],
)
# Log the information
logging.info(f"Used Weight File: {weight_file}")

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


# # Set up data loaders
test_data = Covid(
    "data/Infection Segmentation Data/Infection Segmentation Data", mode="test"
)

val_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=2)


# for batch_idx, (
#     inputs,
#     labels_classification,
#     labels_segmentation_lungs,
#     labels_segmentation_infected,
# ) in enumerate(val_loader):
#     # To device
#     inputs = inputs.to(device)
#     labels_classification = labels_classification.to(device)
#     labels_segmentation_infected = labels_segmentation_infected.to(device)
#     labels_segmentation_lungs = labels_segmentation_lungs.to(device)

#     (
#         outputs_classification,
#         outputs_segmentation_lungs,
#         outputs_segmentation_infected,
#     ) = model(inputs)

#     outputs_classification = outputs_classification.type(torch.float32)
#     outputs_segmentation_infected = outputs_segmentation_infected.type(torch.float32)
#     outputs_segmentation_lungs = outputs_segmentation_lungs.type(torch.float32)

#     labels_classification = labels_classification.type(torch.float32)
#     labels_segmentation_infected = labels_segmentation_infected.type(torch.float32)
#     labels_segmentation_lungs = labels_segmentation_lungs.type(torch.float32)

#     #             print(outputs_classification ,labels_classification)

#     #         loss_classification = classification_loss_fn(outputs_classification, labels_classification)
#     #         loss_segmentation_infected = segmentation_loss_fn(outputs_segmentation_infected, labels_segmentation_infected)
#     #         loss_segmentation_lungs = segmentation_loss_fn(outputs_segmentation_lungs, labels_segmentation_lungs)
#     # #         loss = (1/3 * loss_classification) + (1/3 * loss_segmentation_infected) + (1/3 * loss_segmentation_lungs)
#     #         loss = (1/3 * loss_classification) + (1/3 * loss_segmentation_infected) + (1/3 * loss_segmentation_lungs)
#     #         val_loss += loss.item() * inputs.size(0)

#     outputs_classification = outputs_classification.argmax(1).detach().cpu().numpy()
#     outputs_segmentation_infected = outputs_segmentation_infected.argmax(1)
#     outputs_segmentation_lungs = outputs_segmentation_lungs.argmax(1)

#     labels_classification = labels_classification.argmax(1).detach().cpu().numpy()
#     labels_segmentation_infected = labels_segmentation_infected.argmax(1)
#     labels_segmentation_lungs = labels_segmentation_lungs.argmax(1)

#     (
#         pixel_acc_infected,
#         dice_infected,
#         iou_infected,
#         precision_infected,
#         recall_infected,
#     ) = calculate_overlap_metrics(
#         labels_segmentation_infected, outputs_segmentation_infected, eps=1e-5
#     )
#     (
#         pixel_acc_lungs,
#         dice_lungs,
#         iou_lungs,
#         precision_lungs,
#         recall_lungs,
#     ) = calculate_overlap_metrics(
#         labels_segmentation_lungs, outputs_segmentation_lungs, eps=1e-5
#     )
#     precision_classification = precision_score(
#         labels_classification, outputs_classification, average="macro"
#     )
#     recall_classification = recall_score(
#         labels_classification, outputs_classification, average="macro"
#     )
#     f1_score_classification = f1_score(
#         labels_classification, outputs_classification, average="macro"
#     )

#     pixel_acc_infected_meter.update(pixel_acc_infected, inputs.shape[0])
#     dice_infected_meter.update(dice_infected, inputs.shape[0])
#     iou_infected_meter.update(iou_infected, inputs.shape[0])
#     precision_infected_meter.update(precision_infected, inputs.shape[0])
#     recall_infected_meter.update(recall_infected, inputs.shape[0])

#     pixel_acc_lungs_meter.update(pixel_acc_lungs, inputs.shape[0])
#     dice_lungs_meter.update(dice_lungs, inputs.shape[0])
#     iou_lungs_meter.update(iou_lungs, inputs.shape[0])
#     precision_lungs_meter.update(precision_lungs, inputs.shape[0])
#     recall_lungs_meter.update(recall_lungs, inputs.shape[0])

#     precision_classification_meter.update(precision_classification, inputs.shape[0])
#     recall_classification_meter.update(recall_classification, inputs.shape[0])
#     f1_score_classification_meter.update(f1_score_classification, inputs.shape[0])
# #             f1_score(y_true, y_pred, average='macro')

# f1_score_infected_meter = calculate_f1_score(
#     precision_infected_meter.avg, recall_infected_meter.avg
# )
# f1_score_lungs_meter = calculate_f1_score(
#     precision_lungs_meter.avg, recall_lungs_meter.avg
# )

# logging.info(
#     f"pixel_acc_infected: {pixel_acc_infected_meter.avg :.4f}, dice_infected: {dice_infected_meter.avg :.4f},iou_infected: {iou_infected_meter.avg :.4f}, precision_infected: {precision_infected_meter.avg :.4f}, recall_infected: {recall_infected_meter.avg :.4f}, f1_score_infected: {f1_score_infected_meter :.4f} \n \
# pixel_acc_lungs: {pixel_acc_lungs_meter.avg :.4f}, dice_lungs: {dice_lungs_meter.avg :.4f},iou_lungs: {iou_lungs_meter.avg :.4f}, precision_lungs: {precision_lungs_meter.avg :.4f}, recall_lungs: {recall_lungs_meter.avg :.4f}, f1_score_lungs: {f1_score_lungs_meter :.4f} \n\
#     precision_classification: {precision_classification_meter.avg :.4f}, recall_classification: {recall_classification_meter.avg :.4f},f1_score_classification: {f1_score_classification_meter.avg :.4f} \n"
# )


# pixel_acc_infected_meter = AverageMeter()
# dice_infected_meter = AverageMeter()
# iou_infected_meter = AverageMeter()
# precision_infected_meter = AverageMeter()
# recall_infected_meter = AverageMeter()

# pixel_acc_lungs_meter = AverageMeter()
# dice_lungs_meter = AverageMeter()
# iou_lungs_meter = AverageMeter()
# precision_lungs_meter = AverageMeter()
# recall_lungs_meter = AverageMeter()

# precision_classification_meter = AverageMeter()
# recall_classification_meter = AverageMeter()
# f1_score_classification_meter = AverageMeter()


with torch.no_grad():
    for i in tqdm(range(len(test_data))):
        (
            input,
            labels_classification,
            labels_segmentation_lungs,
            labels_segmentation_infected,
        ) = test_data[i]
        inputs = input.unsqueeze(0).to(device)
        labels_classification = labels_classification.unsqueeze(0)
        labels_segmentation_lungs = labels_segmentation_lungs.unsqueeze(0)
        labels_segmentation_infected = labels_segmentation_infected.unsqueeze(0)
        # inputs = input.unsqueeze(0)

        # for batch_idx, (inputs, labels_classification,  labels_segmentation_lungs, labels_segmentation_infected) in enumerate(val_loader):
        #     # To device
        #     print('ss')
        # inputs = inputs.to(device)
        labels_classification = labels_classification.to(device)
        labels_segmentation_infected = labels_segmentation_infected.to(device)
        labels_segmentation_lungs = labels_segmentation_lungs.to(device)

        (
            outputs_classification,
            outputs_segmentation_lungs,
            outputs_segmentation_infected,
        ) = model(inputs)

        outputs_classification = outputs_classification.argmax(1).detach().cpu().numpy()
        # print(outputs_classification)

        outputs_segmentation_infected = outputs_segmentation_infected.argmax(1)
        outputs_segmentation_lungs = outputs_segmentation_lungs.argmax(1)

        labels_classification = labels_classification.argmax(1).detach().cpu().numpy()
        # print(labels_classification)
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

        precision_classification_meter.update(precision_classification, inputs.shape[0])
        recall_classification_meter.update(recall_classification, inputs.shape[0])
        f1_score_classification_meter.update(f1_score_classification, inputs.shape[0])

    mem = "%.3gG" % (
        torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    )  # (GB)
    logging.info(mem)
#             f1_score(y_true, y_pred, average='macro')
# val_loss /= len(val_loader.dataset)
# scheduler.step(val_loss)
# print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} \n \
f1_score_infected_meter = calculate_f1_score(
    precision_infected_meter.avg, recall_infected_meter.avg
)
f1_score_lungs_meter = calculate_f1_score(
    precision_lungs_meter.avg, recall_lungs_meter.avg
)
logging.info(
    f"pixel_acc_infected: {pixel_acc_infected_meter.avg :.4f}, dice_infected: {dice_infected_meter.avg :.4f},iou_infected: {iou_infected_meter.avg :.4f}, precision_infected: {precision_infected_meter.avg :.4f}, recall_infected: {recall_infected_meter.avg :.4f}, f1_score_infected: {f1_score_infected_meter :.4f} \n \
pixel_acc_lungs: {pixel_acc_lungs_meter.avg :.4f}, dice_lungs: {dice_lungs_meter.avg :.4f},iou_lungs: {iou_lungs_meter.avg :.4f}, precision_lungs: {precision_lungs_meter.avg :.4f}, recall_lungs: {recall_lungs_meter.avg :.4f}, f1_score_lungs: {f1_score_lungs_meter :.4f} \n\
    precision_classification: {precision_classification_meter.avg :.4f}, recall_classification: {recall_classification_meter.avg :.4f},f1_score_classification: {f1_score_classification_meter.avg :.4f} \n"
)


# logging.info(f"{f1_score_classification_meter.avg}, {f1_score_lungs_meter}, {iou_lungs_meter.avg}, {dice_lungs_meter.avg}")
# logging.info(f"{f1_score_infected_meter}, {iou_infected_meter.avg}, {dice_infected_meter.avg}")
# logging.info(f"{(f1_score_classification_meter.avg + f1_score_infected_meter + f1_score_lungs_meter) / 3}")

export_to_excel(
    'results/w_o_post_processing.xlsx',
    [
        "w/o post processing",
        f"{custom_model.encoder_name}_{custom_model.decoder_name}",
        f1_score_classification_meter.avg,
        f1_score_lungs_meter,
        iou_lungs_meter.avg,
        dice_lungs_meter.avg,
        f1_score_infected_meter,
        iou_infected_meter.avg,
        dice_infected_meter.avg,
        (
            f1_score_classification_meter.avg
            + f1_score_infected_meter
            + f1_score_lungs_meter
        )
        / 3,
        count_parameters(model)
    ],
)

logging.info("after postprocess")

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

with torch.no_grad():
    for i in tqdm(range(len(test_data))):
        (
            input,
            labels_classification,
            labels_segmentation_lungs,
            labels_segmentation_infected,
        ) = test_data[i]
        inputs = input.unsqueeze(0).to(device)
        labels_classification = labels_classification.unsqueeze(0)
        labels_segmentation_lungs = labels_segmentation_lungs.unsqueeze(0)
        labels_segmentation_infected = labels_segmentation_infected.unsqueeze(0)
        # inputs = input.unsqueeze(0)

        # for batch_idx, (inputs, labels_classification,  labels_segmentation_lungs, labels_segmentation_infected) in enumerate(val_loader):
        #     # To device
        #     print('ss')
        # inputs = inputs.to(device)
        labels_classification = labels_classification.to(device)
        labels_segmentation_infected = labels_segmentation_infected.to(device)
        labels_segmentation_lungs = labels_segmentation_lungs.to(device)

        (
            outputs_classification,
            outputs_segmentation_lungs,
            outputs_segmentation_infected,
        ) = model(inputs)

        outputs_segmentation_lungs = (
            np.transpose(
                outputs_segmentation_lungs.argmax(1).detach().cpu().numpy(), (1, 2, 0)
            )
            * 255
        ).astype("uint8")
        outputs_segmentation_infected = (
            np.transpose(
                outputs_segmentation_infected.argmax(1).detach().cpu().numpy(),
                (1, 2, 0),
            )
            * 255
        ).astype("uint8")
        # print
        _, outputs_segmentation_lungs, outputs_segmentation_infected = post_processing(
            outputs_classification,
            outputs_segmentation_lungs,
            outputs_segmentation_infected,
        )
        outputs_segmentation_lungs = np.expand_dims(outputs_segmentation_lungs, axis=2)
        outputs_segmentation_infected = np.expand_dims(
            outputs_segmentation_infected, axis=2
        )

        # print(np.unique(outputs_segmentation_lungs))
        # plt.imshow(outputs_segmentation_lungs,cmap='gray')
        # cv2.imwrite('outputs_segmentation_lungs.jpg',outputs_segmentation_lungs)

        outputs_classification = outputs_classification.argmax(1).detach().cpu().numpy()
        # outputs_segmentation_infected = outputs_segmentation_infected.argmax(1)
        # outputs_segmentation_lungs = outputs_segmentation_lungs.argmax(1)

        labels_classification = labels_classification.argmax(1).detach().cpu().numpy()
        labels_segmentation_infected = (
            np.transpose(
                labels_segmentation_infected.argmax(1).detach().cpu().numpy(), (1, 2, 0)
            )
            * 255
        ).astype("uint8")
        labels_segmentation_lungs = (
            np.transpose(
                labels_segmentation_lungs.argmax(1).detach().cpu().numpy(), (1, 2, 0)
            )
            * 255
        ).astype("uint8")
        # print(np.unique(labels_segmentation_lungs))

        # cv2.imwrite('labels_segmentation_lungs.jpg',labels_segmentation_lungs)

        (
            pixel_acc_infected,
            dice_infected,
            iou_infected,
            precision_infected,
            recall_infected,
        ) = calculate_overlap_metrics_post(
            torch.from_numpy(labels_segmentation_infected),
            torch.from_numpy(outputs_segmentation_infected),
            eps=1e-5,
        )
        (
            pixel_acc_lungs,
            dice_lungs,
            iou_lungs,
            precision_lungs,
            recall_lungs,
        ) = calculate_overlap_metrics_post(
            torch.from_numpy(labels_segmentation_lungs),
            torch.from_numpy(outputs_segmentation_lungs),
            eps=1e-5,
        )
        # print(dice_lungs,iou_lungs, precision_lungs, recall_lungs)
        # break

        f1_score_infected = (
            2
            * precision_infected
            * recall_infected
            / (precision_infected + recall_infected)
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

        precision_classification_meter.update(precision_classification, inputs.shape[0])
        recall_classification_meter.update(recall_classification, inputs.shape[0])
        f1_score_classification_meter.update(f1_score_classification, inputs.shape[0])

#             f1_score(y_true, y_pred, average='macro')
# val_loss /= len(val_loader.dataset)
# scheduler.step(val_loss)
# print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} \n \
f1_score_infected_meter = calculate_f1_score(
    precision_infected_meter.avg, recall_infected_meter.avg
)
f1_score_lungs_meter = calculate_f1_score(
    precision_lungs_meter.avg, recall_lungs_meter.avg
)

logging.info(
    f"pixel_acc_infected: {pixel_acc_infected_meter.avg :.4f}, dice_infected: {dice_infected_meter.avg :.4f},iou_infected: {iou_infected_meter.avg :.4f}, precision_infected: {precision_infected_meter.avg :.4f}, recall_infected: {recall_infected_meter.avg :.4f}, f1_score_infected: {f1_score_infected_meter :.4f} \n \
pixel_acc_lungs: {pixel_acc_lungs_meter.avg :.4f}, dice_lungs: {dice_lungs_meter.avg :.4f},iou_lungs: {iou_lungs_meter.avg :.4f}, precision_lungs: {precision_lungs_meter.avg :.4f}, recall_lungs: {recall_lungs_meter.avg :.4f}, f1_score_lungs: {f1_score_lungs_meter :.4f}  \n\
    precision_classification: {precision_classification_meter.avg :.4f}, recall_classification: {recall_classification_meter.avg :.4f},f1_score_classification: {f1_score_classification_meter.avg :.4f} \n"
)

# logging.info(f"{f1_score_classification_meter.avg}, {f1_score_lungs_meter}, {iou_lungs_meter.avg}, {dice_lungs_meter.avg}")
# logging.info(f"{f1_score_infected_meter}, {iou_infected_meter.avg}, {dice_infected_meter.avg}")
# logging.info(f"{(f1_score_classification_meter.avg + f1_score_infected_meter + f1_score_lungs_meter) / 3}")
export_to_excel(
    'results/w_post_processing.xlsx',
    [
        "w post processing",
        f"{custom_model.encoder_name}_{custom_model.decoder_name}",
        f1_score_classification_meter.avg,
        f1_score_lungs_meter,
        iou_lungs_meter.avg,
        dice_lungs_meter.avg,
        f1_score_infected_meter,
        iou_infected_meter.avg,
        dice_infected_meter.avg,
        (
            f1_score_classification_meter.avg
            + f1_score_infected_meter
            + f1_score_lungs_meter
        )
        / 3,
        count_parameters(model)
    ],
)
