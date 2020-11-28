Mini Project - 2

Rajavi Mehta

001057845

Web-Scraping and Sentimental Analysis on Amazon Reviews

Abstract 

This assignnment includes web-scrapinng of data from a website - amazon.com. After creating the dataset, pre-processing is done and it is fed into the models built, Naive Bayes Classifier, Logistic Regression Classifier, XGBoost, Random Forest Classifier and KNN, which trains on this pre processed data and testing is done on the testing set and accuracy is calculated. To validate the accuracy, validation set is then fed into the models to see if accuracy is appropriate.

Dataset:

Dataset contains 2 columns:

Ratings - Ratings given by the customer to the product

Comments - Reviews of the customers on the product

For detailed implementation information: https://rajavimehta.medium.com/web-scraping-and-sentimental-analysis-on-amazon-reviews-76130bc21463
  
Conclusion: Conclusion: After analysing the above models, we come to a conclusion that XGBoost model gives the best accuracy when compared with the other models at 89.00%. Followed by Random Forest at 88.13%, Multinomial NB at 87.93% and Logistic Regression at 86.47%. The least accuracy is given by KNN at 63.53%. But for the validation set, Multinomial NB given the highest accuracy at 89.57% followed by XGBoost at 87.97%. Random Forest Classifier and Logistic Regression is almost same at 86.96% and 86.77% respectively. The lowest accuracy for the validation set is given by KNN at 81.96%

Citations:

https://www.kaggle.com/him4318/sentimental-analysis

https://www.youtube.com/watch?v=ve_0h4Y8nuI&list=PLhTjy8cBISEqkN-5Ku_kXG4QW33sxQo0t

https://blog.ekbana.com/pre-processing-text-in-python-ad13ea544dae

https://www.analyticssteps.com/blogs/introduction-natural-language-processing-text-cleaning-preprocessing

https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis


License Copyright 2020 Rajavi Mehta

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
