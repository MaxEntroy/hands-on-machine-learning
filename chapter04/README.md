## chapter04 线性回归

## Basic

- total sample = feature + tag
- feature: Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population
- tag: Price.

我们的任务是学习到tag和features的关系。

- 目前来看，推荐系统的sample，行为特征类似click, view。这个到底算什么？
- 因为看score算是最终的tag。类比这里的price 但问题是，那click, view这些又算什么？

刚想明白了：
- total sample = feature + tag
- feature: user features + item features + context features.
- tag: ctr
- 这里的问题是，total sample = feature + tag，这个tag应该是ctr。那这个东西从哪拿呢？
- ctr = click / view. click/view是用户的行为，所以对于total sample来说ctr是tag，没问题。
- 但是它以另一种形式体现出来。所以，推荐系统里面也说tag是click/view这些东西。