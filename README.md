### Command to reproduce results :
   ```
   python main.py --dataroot=<dataroot> --dataset=<dataset> --method=<method> --num_epochs=<epochs> --epoch_decay_start=<eds> --epsilon=<eps> --top_bn=<top_bn>
   ```
### Arguments and Possible Inputs:

1. **`--dataroot=<dataroot>`**
   - **Purpose**: Specifies the root directory where the dataset is stored.
     - Any valid directory path (e.g., `/path/to/data` or `C:/datasets`).

2. **`--dataset=<dataset>`**
   - **Possible Inputs**: 
     - `svhn` (Street View House Numbers dataset)
     - `cifar10` (CIFAR-10 dataset)
     - `mnist` (MNIST dataset)

3. **`--method=<method>`**
   - **Purpose**: Specifies the method to use for training.
   - **Possible Inputs**: 
     - `vat` (Virtual Adversarial Training)
     - `vatent` (VAT with Entropy loss)

4. **`--num_epochs=<num_epochs>`**
   - **Purpose**: Specifies the total number of training epochs.
     
5. **`--epoch_decay_start=<epoch_decay_start>`**
   - **Purpose**: Specifies the epoch at which learning rate decay should start.

6. **`--epsilon=<epsilon>`**
   - **Purpose**: Specifies the hyperparameter controlling the strength of the adversarial perturbation.

7. **`--top_bn=<top_bn>`**
   - **Purpose**: Specifies whether to use batch normalization in the top layers of the model.
   - **Possible Inputs**: 
     - `True` 
     - `False`

---

### Command Summary with Arguments and Examples:

1. **MNIST with VAT loss:**
   ```
   python main.py --dataroot=<dataroot> --dataset=mnist --method=vat
   ```


4. **CIFAR10 with VAT + Entropy loss:**
   ```
   python main.py --dataroot=<dataroot> --dataset=cifar10 --method=vatent --num_epochs=500 --epoch_decay_start=460 --epsilon=10.0 --top_bn=False
   ```
