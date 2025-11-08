## Dataset

https://www.kaggle.com/competitions/nlp-getting-started/data?selb ect=train.csv

## Topic

See the differences in performance between:

(Crista)
- TF-IDF + Logistic Regression
    ```python

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    # from train.csv
    X_train = ["I love this!", "This is terrible", "Amazing product", "Worst experience ever"] # text tweet array
    y_train = [1, 0, 1, 0] # target array corresponding to the tweets

    # from test.csv
    X_test = ["Great stuff", "Not good at all"]
    y_test = [1, 0]

    # TODO: consult the hyperparams
    # NOTE: play with the hyperparameters
    vectorizer = TfidfVectorizer(
        max_features=5000,  # limit vocabulary size
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,  # ignore terms that appear in less than 2 documents
        max_df=0.8  # ignore terms that appear in more than 80% of documents
    )

    # train the vectorizer and get the rezult, for train set
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # get the result from vectorizer only, for test set
    X_test_tfidf = vectorizer.transform(X_test)

    # TODO: consult the hyperparams
    # NOTE: play with the hyperparameters (search the hyperparameters recomended for the models)
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # useful for imbalanced datasets
    )

    # train the model, on train set
    model.fit(X_train_tfidf, y_train)

    # get prediction, for test set
    y_pred = model.predict(X_test_tfidf)
    # output: [0, 1, 0, 1, ...]

    #TODO: 
    #    - consult what is the input and output for the models
    #    - consult what the models do, and explain a bit the process
    #    - metrics: accuracy, precision and recall
    #    - save the predictions (ask me)
    ```

    - see what the model is using internally and document it in the notebook a bit (or paper)
    <!-- - after obtaining good trainings: do a for loop and train 4-5 times to see the measurements are consistent, keep them in a list or dict  (ask me if this remains) -->

    Papers for the models:
    1. TF-IDF
        Original TF-IDF concept:

            Spärck Jones, K. (1972). "A statistical interpretation of term specificity and its application in retrieval." Journal of Documentation, 28(1), 11-21.

        TF-IDF + Logistic Regression for text classification:

            Das, M., & Chakraborty, S. (2023). "A Comparative Study on TF-IDF feature Weighting Method and its Analysis using Unstructured Dataset." arXiv preprint arXiv:2308.04037.

    2. BERTweet

            Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). "BERTweet: A pre-trained language model for English Tweets." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 9-14).

    3. Llama 3.2

            Dubey, A., Jauhri, A., Pandey, A., et al. (2024). "The Llama 3 Herd of Models." arXiv preprint arXiv:2407.21783.

    TODO: present the architectures of the two models;

    Papers done before with this task (Kaggle Competition & Disaster Tweets):

    1. RIT Thesis on disaster tweets:

            Alhammadi, H. (2022). "Using Machine Learning in Disaster Tweets Classification." Master's Thesis, Rochester Institute of Technology.


    2. NLP with disaster tweets analysis:

            Published work on the Kaggle NLP disaster tweets dataset from Tennessee Tech (2020): "NLP Disaster Tweets Classification." Proceedings of Student Research and Creative Inquiry Day.

    TODO:
    - for the papers with the concept, summarize what was studies in them and what are the results. Say what we bring new to the picture.
    - present the dataset and examples from it and cite it


(Sonia)
- BERTweet
- Llama 3.2

## Questions

- Does domain-specific pre-training on Twitter data improve disaster detection compared to general-purpose language models?
    - Performance measurements: accuracy, precision, recall 
    - Performance validation: measurement comparison; run the training multiple times to see if the difference is significant or random (Bootstrap Confidence Intervals / Multiple Runs with Different Seeds)
    - Error analysis: What types of tweets does BERTweet handle better? (analyze the false negatives and false positives and see what characteristisc they have - see if there is a correlation)
    - Is the computational cost of transformers justified compared to classical ML (bertweet vs tf-idf)? 
        - training time vs final evalutation

## Steps

- analyze the dataset and preprocess it (remove duplicates, remove any unmeaningfull data (links))
- prepare the data for the training (split the training set)
- train the models (binary cross entropy, take the hyperparameters recomended for the models (batch size, lr, epochs), adam optimizer, linear with warmup scheduler)
- for 4 to 5 training sessions -> see if the result is consistent
- results analysis: see the common vs difference cases (where models agree); create some features for the tweets and see which are favorable for which model, statistical difference (t-test p-value), cofusion matrix, Comparison table