# Testing the baseline: raw model performance, with no measures to help imbalance
python main.py --name baseline
Start-Sleep -Seconds 300

# Loss Level change to address imbalance
python main.py --name baseline_with_weighted_loss --class_weights 
Start-Sleep -Seconds 300

python main.py --name baseline_focal_loss --focal_loss
Start-Sleep -Seconds 300

# Model Architecture level changes to deal with Inter-class Similarity and Intra-Class Variance
python main.py --name baseline_with_weights_dropout --class_weights --dropout

# Data Level Changes to help - Imbalanced data means the model sees more images of one class than another..does making the model see different images make the model generalize better to these classes
#     - Rotation
      python main.py --name rotation_5 --rotate 5 --class_weights --dropout
      Start-Sleep -Seconds 300

      python main.py --name rotation_10 --rotate 10 --class_weights --dropout
      Start-Sleep -Seconds 300

      python main.py --name rotation_45 --rotate 45 --class_weights --dropout
      Start-Sleep -Seconds 300

#     - Flipping
     python main.py --name horizontal_flip --hflip 0.5 --class_weights --dropout
     Start-Sleep -Seconds 300
     python main.py --name vertical_flip --vflip 0.5 --class_weights --dropout
     Start-Sleep -Seconds 300

  # Photometric Transformations: changing contrast
    python main.py --name jitter --jitter 0.3 0.3 --class_weights --dropout
    Start-Sleep -Seconds 300

    python main.py --name clahe --clahe --class_weights --dropout
    Start-Sleep -Seconds 300








