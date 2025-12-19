from torch import tensor, float32
def calculate_mean_standard_dev(dataloader):
    num_batches = len(dataloader)
    sum_of_pixels = 0
    print(f"num batches: {num_batches}")


    # go through every batch in the dataloader 
    for batch_num, (train_features, train_labels) in enumerate(dataloader):

        batch_size = len(train_features)
        # store a sum of the mean for the batch
        batch_mean_sum = 0

        # testing with the first batch only for now
        if batch_num != 0:
            break
        print(f"batch number: {batch_num}")
        print(f"train features size: {train_features.size()}") # [batch_size, num channels, H, W]


        # for each image tensor in the batch - each index is a new image
        # loop through, get the image, and calculate the mean of its pixel values
        # and add it to the sum for the batch, and divide by batch size to get mean
        for i in range(batch_size):
            print(f"train features: {train_features[i].size()}") # a 224 x 224 grayscale img
            print(f"image number: {i + 1}") # index starts at 0
            image = train_features[i].squeeze()
            print(f"IMAGE TENSOR: {image.size()}")
            image = image.to(dtype=float32)
            image_mean = image.mean()
            print(f"image mean = {image_mean}")
            batch_mean_sum += image_mean


        print(f"sum of means for the first batch = {batch_mean_sum}")
        batch_mean = batch_mean_sum / batch_size
        print(f"the mean for all images in the first batch is {batch_mean}")
