def getEmbedding(all_cleaned_texts, tokenizer, mode = "GN300"):
  import multiprocessing
  import gensim
  import gensim.downloader as api
  import tensorflow_hub as hub
  import numpy as np

  dim = 300 # Dimentions of the embedding matrix

  if mode == "customTrained":
    modelW2V = gensim.models.Word2Vec(all_cleaned_texts, size= dim, min_count = 2, window = 5, sg=0, iter = 10, workers= multiprocessing.cpu_count() - 1)
  elif mode == "GN300":
    word2vec = api.load("word2vec-google-news-300")
    dim = 300
  else:
    word2vec = hub.load("https://tfhub.dev/google/nnlm-en-dim128/2")
    # dim = 128
    dim = 200   # Even though NNLM has 128 dimentions, in some runs, we observed that given a little space for fine tuning at model level gives a flexibility with the cost of overfitting

  num_words = len(list(tokenizer.word_index))

  embedding_matrix = np.random.uniform(-1, 1, (num_words + 1, dim))
  for word, i in tokenizer.word_index.items():
      if i < num_words:
        try:
          embedding_vector = word2vec.get_vector(word)
          if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
          else:
            embedding_matrix[i] = np.zeros((dim,))
        except:
          pass

  embedding_matrix[num_words] = np.zeros((dim,))
  return embedding_matrix