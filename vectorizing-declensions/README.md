## Vectorizing Declensions - On generating embeddings for declension-intensive languages

1. Analysis on processing languages that make extensive use of declensions: creating good embeddings is specially difficult but still critical
2. Experiments on how such embeddings could be created without multiplying the requirements on corpora size and processing power.

## Brief intro to word embeddings 

When applying Deep Learning for Natural Language Processing, state of the art techniques rely on the following process:
1. Convert individual words into embeddings (high-dimensional vectors). This can be done using Word2Vec or Glove
2. Feed those list of embeddings into neural networks
A great explanation on this can be found at http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/

In order to generate the word embeddings, we need to build a "word-to-vector" mapping. Not any mapping is desirable, but we want it to be efficient (similar words should be mapped to similar vectors). State of the art techniques to do so consist in taking a large text corpora and apply a specific algorithm on it. The two main algorithms to do that are: 
a. Word2Vec (http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
b. GloVe (https://nlp.stanford.edu/pubs/glove.pdf)

Both algorithms can provide great results, but that depends not only on the algorithm, but also on the text corpora we use as input to the algorithm. When it comes to getting a text corpora that will be enough to generate good embeddings, the language becomes very relevant.

## Declension-intensive languages and embeddings created from text corpora

In text corpora we can find different types of words: nouns, pronouns, verbs, adjectives, adverbs, prepositions, articles, determiners...

But when it comes to "how many words of each type", there are important differences among languages.

To construct a sentence, some languages will use a noun (or pronoun) together with articles, prepositions and determiners
Other languages do not add those articles, prepositons or determiners, but just transform the noun (or pronoun) so that the now transformed (declensed) word contains all the information. This way much information is condensed in one word. This has several implications:
 * Now for each noun and pronoun, a lot of different transformed versions (declensions) are possible, which results in a huge vocabulary
 * However, as sentences are now shorter (much meaning in few words), trying to understand the meaning of each word by looking at the other words in the same sentence is harder (as there are few other words in the same sentence). This is bad for Word2Vec and GloVe algorithms, as they just analyse each word with its context (other words in same sentence).
 * And getting the meaning of each word right is critical, as it contains much information.

Must be said that applying deep-learning on declension intensive-languages could present additional difficulties if such languages are also ergative languages, whose morphosyntactic allignment is different from that of English and most other European languages (nominative-accusative languages). It makes sense that making a deep learning system to work with two different morphosyntatic allignments at the same time should be more complex. Must be said that, when doing this, getting declensions right is again critical. More info on ergative vs accusative on https://en.wikipedia.org/wiki/Ergative%E2%80%93absolutive_language

Thus, we can conclude that, for declension-intensive languages, creating good embeddings from text corpora will be more difficult, but still very important. So how to do it? The next section will describe some ideas to do so, and the purpose of this project is to experiment on them

## Looking for an efficient way to create word embeddings for declension intensive languages

As discussed previously, we have huge vocabulary size (many declensed versions of each word) and few words per sentence. We could think in several solutions to target this:

1. The conventional way to target this problem would be to use much larger (than for other languages) text corpora. However, this is obviously more expensive.

2. We could also think about several stages using different types of text corpora. At each stage, we use a feature-specific text corpora, so we generate vectors that are also feature specific (only the dimensions corresponding to those features). For instance, we can do the following:
 2.1. Create a declension-specific text corpora, which we will use for Word2Vec to get the relationships (dimensions in the output vectors) between each "undeclensed" noun/pronoun and its corresponding declensions. This is tested in word2vec-on-tagged-declensions-corpora.ipynb (https://github.com/fran-ovia/deep-learning-experiments/blob/master/vectorizing-declensions/word2vec-on-tagged-declensions-corpora.ipynb)
 2.2. After that, we get general corpora, undeclense all the nouns/pronouns (which can be done with the model calculated in previous step) and run Word2Vec on it, so the vectors at our output will get everything right except the declensions (*). Declension dimensions are not present in these vectors, however, we can bring those dimensions back from our previous step on declension-specific text corpora. (*) **However, there are problems with this: by just undeclensing words before running Word2Vec we are removing much information (dimensions) which will affect negatively the performance of the embeddings generated by Word2Vec, in a way that the damage could not be completely repaired by just adding back the dimensions later** (Word2Vec would get relationships wrong, unless the dimensions generated by Word2Vec in the new step are completely orthogonal to those we have previously removed by undeclensing, but it is not realistic to rely on that). Thus, this is not really a good method.

3. A slight variant to solution 2. Same idea but dimensions and values specific to declensions (step 2.1) are generated by not statistical but analytical methods, as the grammatical rules for the declensions can be codified at a moderate cost, whereas generating corpora with all the possible declensions for all the words can be very expensive. Additionally, step 2.1 had other problems (relatively small corpora, with each unique declensed word appearing only once requires weird tuning). However, problem described at step 2.2. is still present and is a big issue, so this is still not a good method.

4. Several stages with different feature-specific text corpora, but the vectors we generate in previous stages are used as "embeddings" to preprocess a next text corpora. The idea here is to (at previous stages) translate each input word to an output vector which will then be mapped to a sequence of words consisting on the "originally undeclensed" word followed by a list of dimension tags (dimension tags would represent grammatical cases). Then in the final text corpora, we would replace each word by the corresponding "undeclensed word and dimension tags" that can be mapped by using the models from previous steps, so we can then run Word2Vec without losing information.

5. A slight variant to solution 4. Same idea but the mappings (word to "originally undeclensed" word followed by a list of dimension tags) to generate at previous stages are not generated by using feature-specific text corpora, but by analytical methods. This way, we just run run an analytical preprocess on the final text corpora to be vectorized with Word2Vec. Such "analytical preprocess" would just consist on mapping each word to the corresponding "undeclensed word" followed by a list of grammatical case tags. This seems to be a more cost-effective approach to that in solution 4 (as the grammatical rules for the declensions can be codified at a moderate cost, whereas generating corpora with all the possible declensions for all the words can be very expensive), and still avoids the dimension orthogonality problems with solutions 2 and 3.

(I must apologize for this section being written as just a ramble on the possible solutions. I'm planning to rewrite it to a clearer format soon).

## Acknowledgements

Thanks to:
 * Radim Řehůřek Ph.D for open sourcing an implementation of Word2Vec (https://radimrehurek.com/gensim/models/word2vec.html)
 * Siraj Raval for sharing code we could use as an example: https://github.com/llSourcell/word_vectors_game_of_thrones-LIVE

