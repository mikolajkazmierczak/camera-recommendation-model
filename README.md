# camera-recommendation-model

[Model rekomendacji aparatów fotograficznych.pdf](https://github.com/mikolajkazmierczak/camera-recommendation-model/blob/main/Model%20rekomendacji%20aparat%C3%B3w%20fotograficznych.pdf)

**Note:** The whole report with the details of the project can be found in the pdf file above (that is also in the repository).

This project introduces a **camera recommendation model that analyzes user reviews using NLP**, aggregates component ratings into weighted scores, and uses cosine similarity to personalize recommendations, including handling cold-start users through demographic clustering. A prototype implementation is available in this repo. The system prioritizes features most valued by users for tailored suggestions.

An excerpt from the report:\
"The implementation of this project allowed us to better understand the issue of user profiling in recommendation systems, as well as the multitude of problems during the design of such a system. Since the developed system operated on component ratings determined from reviews, we had to adapt it in order to create aggregation of component ratings into one "holistic" numerical value. Thanks to this, however, the model recommends a product with features that the interested user values ​​the most."
