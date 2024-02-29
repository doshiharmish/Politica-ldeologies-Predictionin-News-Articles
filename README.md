# Political Ideologies Prediction in News Articles

A project analyzing and predicting political biases in news articles with a focus on providing real-time, lightweight insights for media literacy.

## Overview

In the realm of today's diverse media landscape, it's increasingly important to critically evaluate news content. Our project aims to contribute to media literacy by predicting political bias in news articles, assisting readers in understanding subjective elements present within the content. Featuring a PySpark-driven inference pipeline, Natural Language Processing, and Machine Learning models, this project is versatile across platforms.  To demonstrate functionality, we've integrated it with the well-regarded New York Times API enabling real-time bias prediction within top political stories.

## Business Goals

* **Enhanced Media Literacy:** Provide readers with a tool to disclose potential political leanings in news articles, encouraging informed consumption.
* **Policy and Regulation Insights:** Shed light on biased reporting during elections, potentially informing media guidelines to guarantee fairness and integrity.

## Prediction Goals

* **Effective Bias Classification:** Develop a model skilled at classifying political bias in news articles (left, center, right) based on their content. 
* **Real-Time Detection:** Apply the model for immediate bias detection within newly released articles from top news sources like the New York Times.
* **Understanding Content-Based Bias:** Uncover patterns and language cues within news articles that suggest specific political inclinations.

## Dataset Description

Our dataset, obtained from the WebIS research group's GitHub repository, comprises 7,775 news articles in JSON format, each with 8 attributes. The key attributes are:

* **Allside_bias (Target):** Represents political bias (left, center, right).
* **Content (Predictor):** Article text for analysis.

## Exploratory Data Analysis (EDA) Insights

* **Bias Distribution:** Dataset skewed towards left-leaning content (47.4%).
* **Author Bias:**  Most authors were consistently associated with one specific ideology. While this feature could aid in bias prediction, it introduces the cold-start problem, which poses challenges for model accuracy when encountering new authors.
* **Word Clouds:** Visualization using word clouds was limited in its ability to reveal nuanced linguistic biases.

## Methodology

1. **Data Preprocessing (PySpark, Spark NLP):**
    * Tokenization, Lowercasing Text, Stop Word Removal and more.
2. **Vectorization:**
    * **TF-IDF:** Context-independent, bag-of-words technique. 
3. **Models (Spark MLlib):**
    * Multinomial Naïve Bayes (baseline)
    * Random Forest Classifier
4. **Evaluation:** F1-score prioritized due to the multi-class target variable.
5. **NYT API Integration:** Real-time article analysis.


## Insights Gained

* **Confirmation of Author Bias:** Our analysis confirmed that most authors predominantly publish articles aligned with one particular political orientation. This underscores the importance of considering the author's history when assessing potential bias.  
* **Challenges in Identifying Centrist Content:** The models struggled to accurately classify 'center' labeled articles, indicating the difficulty in identifying content with a balanced or neutral perspective. 
* **Keywords Reveal Sentiment:**  Top features for each classification included keywords that aligned with expected political associations (e.g., "Obama," and "Sanders" for left-leaning articles).

## Results

* **Random Forest Classifier with TF-IDF vectorization outperformed the Naïve Bayes baseline.** 
* **Feature importance analysis** provided insights into keywords associated with each bias classification.
* **Challenges in identifying 'center' labeled articles** highlight the complexity of unbiased content.

**Model Performance with TF-IDF Vectorization**

| Model                | Accuracy | Precision (left, right, center) | Recall (left, right, center) | F1-Macro Score |
| -------------------- | -------- | ------------------------------- | ---------------------------- | -------------- |
| Naïve Bayes          | 60.48%   | 70%, 66%, 34%               | 61%, 62%, 52%                | 61.45%         |
| Random Forest        | 68.24%   | 60%, 86%, 97%               | 94%, 58%, 16%                | 65.08%         |

* **Class-Specific Insights:**  
    * **Left:** Both models showed higher recall than precision for the "left" class, indicating they were good at finding left-leaning articles but sometimes misclassified other articles as "left."

    * **Right:**  The Naive Bayes model had similar precision and recall for the "right" class. In contrast, the Random Forest model had notably higher precision, meaning it rarely misclassified articles as "right."

    * **Center:** Both models had the lowest scores for the "center" class, highlighting the challenge of  detecting neutral or unbiased news content. 

## Demonstration (NYT API)

The best-performing model was applied to top stories from the New York Times. While ground truth labels were unavailable, a manual review was conducted.  As anticipated, nuanced categories, especially 'center', proved more challenging to classify.

## Future Scope

* **Diverse Dataset:** Collect more varied and shorter documents for a more diverse training set.
* **Context-Based Vectorization:** Explore BERT/ROBERTA models for advanced contextual analysis.
* **Advanced Techniques:** Experiment with SVMs and ANNs to potentially enhance prediction accuracy.

## Conclusion

This project demonstrates the ability to predict political bias in news articles. Our TF-IDF Random Forest model and NYT API integration show potential for real-time applications. This effort contributes to the discussion surrounding media literacy and responsible news consumption. 

## References
1.	Beutel, Alexander, et al. "Analyzing Political Bias and Unfairness in Language Models: A Case Study of GPT-3." Paper with Code, 2023. https://paperswithcode.com/paper/analyzing-political-bias-and-unfairness-in
2.	Weßler, Christoph, et al. "NLPCSS-20 Dataset and Code." GitHub, 2023. https://github.com/webis-de/NLPCSS-20/tree/main/data
3.	The New York Times. "New York Times Developer API." https://developer.nytimes.com/apis
4.	den Heijer, Michael. "pynytimes." https://github.com/michadenheijer/pynytimes
5.	"Top Stories - pynytimes." https://pynytimes.michadenheijer.com/popular/top-stories
6.	Spark NLP. "Spark NLP for Natural Language Processing." https://sparknlp.org/docs/en/quickstart
7.	Apache Spark. "Machine Learning Guide." https://spark.apache.org/docs/latest/ml-guide.html
8.	Apache Spark. "Spark Python API." https://spark.apache.org/docs/latest/api/python/index.html
9.	Public APIs. "News." Public APIs, 2023. https://github.com/public-apis/public-apis#news 
