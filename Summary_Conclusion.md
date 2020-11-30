## Summary

With the shorter deadline and having a nice, clean dataset that we had already processed in Assignment 2, we iniitally though that this forecasting assignment would be fairly straight forward. We quickly realised that this wouldn't be the case after some initial attempts to do some univariate forecasting. 

The thought process for our initial steps was to take each subject individually and try to forecast how their sleep cycle would develop throughout the night. It dawned on us, however, that we actually had very little data on a "per subect" basis. We only have 1 sleep cycle per person and so any attempt to split that cycle into training/test sets leaves a glaring hole in what the model can actually learn. For example, if you do an 80/20 split with the 20 being the last 20 percent of the cycle, the model can only assume that the person will never wake up as it has never seen a "waking up" event.

This meant that our first Univariate forecasting, using Facebook's time series forecasting package fbprophet, was particularly poor. We saw that what basically happens is the model uses the previous trend to predict where the sleep cycle would go. This isn't an awful idea but we found that it essentially meant our subject would head towards sleep stage 5 and stay there (never waking up) or somewhat rapidly move towards 0 (awake). This doesn't reflect at all how the sleep cycles actually work, with subjects likely to move between various stages throughout the night before they all eventually wake. 

We decided that we would make the assumption that the first 30 subjects sleep cycles were in fact the previous 30 nights sleep for the 31st subject. In effect, creating a month's worth of sleep for a single subject. This allowed us to give actual full cycles to the training model. We found that this improved the model signifcantly as we might expect, though it still wasn't very impressive. We hypothesized that univariate was really limiting us in our training, and that it was time to move on and use the 270+ features that we had extracted from the sleep cycle data in the previous Assignment.

Our first step in this multi variate process to create a baseline prediction performance. We achieved this by splitting our data into training and testing sets, and predicting the labels of future rows based on the last label in the training data. This is obviously quite primitive but allowed us to see if our model could outperform even this basic baseline performance. 

We then tried out a Random Forest Regression model and took note of how it performed, noting that it was an immediate large improvement on this previous baseline. 

Gradient Boosted Trees was the next choice that we went with, it relies on using ensemble techniques to combine weak classifiers - such as decision trees - into a better model. Once more we noted that this improved our error.

In these previous multivariate models, we had been predicting the future sleep stages of all patients whereas in real world applications we would likely only be predicting the sleep of one subject. As in the Univaraite modeling, the training data will never show the model that a patient eventually stops sleeping. Therefore, the model will simply predict that each patient sleeps for as long as it is asked to predict.

To solve this set aside a number of patients as “test” subjects. The rest of the patients were once combined as if they were one, with each sleep cycle following after another.

We created a model for each subject, using the full training data and a portion of the test subject’s sleep data. We thenattempted to predict the rest of the subject’s sleep stages for that night.

We compared the performance of multiple models here and found that the best performing algorithms were Linear Regression and Support Vector Regression. We wanted to do some cross validation next, and though we would like to have chosen one of these two for this process, the time they take was too long. We went with the Random Forest Regressor instead. 

In the previous classification assignemnt, we had noted that the use of the original windowed features provided classifiers that performed almost as well as models trained on extracted features. We decided to do the same here, and examine the performance of forecasting models using only the original features. Once again we compared the performance of multiple algorithms and this time Support Vector Regression edged slightly ahead of Linear Regression as the best choice. This time we chose Random Forest and K Nearest Neighbours for cross validation as once again, support vector regression was far too time consuming and also linear regression doesn not have enough hyperparameters to find.

We saw that models trained using the extracted features dataset generally perform better than those trained on the original features. However, both approaches struggle with forecasting the beginning and end of a sleep cycle. This may be due to the fact that we have concatenated different people’s sleep data together, all with different charactersitics that the model has difficulty consolidating into a recognisable pattern.

It is also interesting to note that different models work better for different subjects. Future work could be done in this area, whereby multiple nights’ sleep are recorded per subject, and a comparison of each model could be carried out to identify a significantly superior algorithm.

## Learning Outcomes

In terms of what we learned throughout the process, once more the assignment proved quite valuable. We approached it initially somewhat naively, and underestimated what would be involved. But through lots of trial and error and a lot of internet searches and banging our heads against the wall we managed to get what we feel are some useful results. It would have been nice to have lots of data for each subject but we had to make do with what we had. As in previous assignments, it seems likely that this will be mirrored somewhat in industry, and that the data that we get to work with will never be perfect for our use case. 

With this in mind, the assignment has undoubtedly aided in preparing us for future tasks both in this final year and once we go into employment. We look forward to whatever challenges assignment 4 might bring. 