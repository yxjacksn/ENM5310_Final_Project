# ENM5310_Final_Project
## Repository Structure

**Models 1 and 2** correspond to the architectures described in the final project report. Within each model directory:
* **`model.py`**: Defines the neural network architecture and model setup.
* **`train.py`**: Contains the training configuration and execution loop.
* **`model.pt`**: Stores the actual trained and saved model weights.

### Evaluation & Results
* **`test_model.py` (Internal Evaluation)**: Evaluates model performance using a 90/10 split on the internal training data to test on unseen samples. However, the results from this evaluation appear artificially inflated due to the data collection methodology. Since each spatial coordinate corresponds to a minimum of 8 images, a random split results in the model encountering test images that are spatially identical to those in the training set, effectively providing the model with significant prior information.
* **`eval.py` (External Evaluation)**: To address the leakage issue, this script evaluates performance using a completely separate dataset that was collected independently and never seen by the model during training.
* **`*.json`**: These files contain the quantitative metrics obtained from the two evaluation methods described above.

The remaining files handle data preprocessing and pipeline utilities.
