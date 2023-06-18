# rec_sys
A recommendation system of scientific articles.
There is a dataset of scientific articles and the task is to create a recommendation system, that takes an article's abstract and returns the value,
that represents the fitting of customer's interest.

I decided to work with article's abstracts because they represents the main idea of the article.
I chose transformer model from https://huggingface.co/sentence-transformers/all-mpnet-base-v2 to get embeddings.
The system decides if recommend an article or not by the cosine similarity of this article with all train articles.
If the mean cosine similarity with 5 top articles is more than threshold, the system recommends this article. 
