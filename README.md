# NBA-Predictions
Python scripts that are designed to identify the top 10 players of an NBA season, as well as conduct match prediction given two teams.

## Abstract
The sporting domain is one of the most widely followed domains in the world. Amongst spectators, the hype before new seasons and global tournaments are unparalleled and leads to deeper analysis into teams, their players as well as their game day strategy. Furthermore, through the advent of social media, it has become increasingly popular for spectators to publicly state their MVP of the season/tournament as well as provide game day predictions. However, when it comes to punters, there is money on the line, and they would want to make informed decisions before placing any bets. Thus, correctly predicting the winning team as well as any outstanding performing players from a season becomes an increasingly difficult problem. Thus, to fill this gap, machine learning techniques can be employed to find complex relationships within the data, which can then be used to make informed decisions. In this paper, we provide two architectures, one for outstanding player detection and the other for game predictions on NBA/ABA data. In particular, to identify outstanding players, we conduct clustering on player data and thereafter select a list of top 10 players through a principled ranking approach. Furthermore, for game predictions, we propose an Attack Defence Win (ADW) Network, which utilises Multi-Layered Perceptron (MLP) and Ensemble techniques to predict the outcome of a game of Basketball from the 2004-2005 NBA season. The proposed models provide adequate results when compared to other models (predictive model) and historical data.


## Introduction
Sport is a widely followed domain in which millions of spectators come together for the passion of the game as well as continuously push their team’s agenda and bragging rights. Throughout history, sporting events have increased in size as well as brough in a large following of new spectators. With the advent of social media, it has become increasingly popular for sporting events to trend on search pages as well as provides the platform where users and pundits share their opinions. Now, more than ever, with the abundance of statistical data, spectators and the likes can make informed predictions on their respective teams. Thus, through machine learning, there is a gap wherein we can apply statistical methodologies as well as appropriate algorithms to learn relationships and trends.

With this implementation, we seek to explore avenues in which we utilise statistical strategies and clustering techniques to find the outstanding players in an ABA/NBA season. Furthermore, we also provide a principled approach in the form of an Attack Defence Win (ADW) Network coupled with ensemble techniques to provide match day predictions for the 2004-2005 NBA season. 


## Proposed Model

### Outstanding Player Detection
The apprach used to detect out standing players follows the following pipeline:
* Data Pre-Processing
* Isolation Forest Anomaly Detection
* Principal Component Analysis (PCA) for visualization
* Ranking


Using PCA for visualization is seen below. The two images depict the selected players before and after ranking.
![2003](https://user-images.githubusercontent.com/24585616/141365030-d094bda8-cce1-4c0c-b6b0-d39f4d23695b.png)
![2003_mod](https://user-images.githubusercontent.com/24585616/141365046-17a8663f-555d-42ec-94f0-6e3ae645b8fc.png)


### Match Prediction
Match predictions were facilitated through an Attack Defense Win Model. The following summary can be used to explain the approach (Refer to the paperfor specific details and application of the models).
* The data is normalised to represent per game stats
* The attacking MLP is trained by using the attacking data
* The defensive MLP is trained using the defensive data
* The win ensemble model is trained by considering offensive points in relation to the defensive points coneded and learning a confidence threshold.

## Code
* Refer to the Deployment Guide for instructions.
