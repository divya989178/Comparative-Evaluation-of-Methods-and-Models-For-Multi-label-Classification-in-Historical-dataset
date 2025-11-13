Two Methods Direct and Indirect Methods

Direct Method: Uses few and zero shot technique through open model (Qwen and Mistral) and Close Models (Gemini and Claude) to annotate the unlabeled dataset.
Indirect Method: Has 4 Main functions(with sub functions): 

*Processor: Prepares the dataset for annotation (tokenization but without attention mask and padding) for training and annotation by GenAI. Creates a file name Cache to store labeled data. It is responsible for updating the cache file with newly annotated samples and reloads the training data with newly annotated samples

*Annotator: Uses both open model and close models to annotate(few shot) desired number of unlabeled samples selected by active learning strategy.

*Active learning strategy plus technique: It is responsible for managing the samples. Selects random samples initialy and sends it to annotator to sample.
Filters out the labeled samples from unlabeled ones and calculates uncertainity of labels using any of the active learning technique (Entropy, KMeansSampling and KNN) 
to get top desired number of samples the model is not sure of and sends it to annotator

*Custom trainer: Model gets trained through GenAI labeled samples

*the main active learning loop: The Main active learning loop which combines all of these methods and runs.
