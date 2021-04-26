library(tidyverse)
library(rpart) 


install.packages('randomForest')
library(randomForest)


# data_url <- getURL(
data <- read.csv('https://raw.githubusercontent.com/TimS-ml/DataMining/master/0_TakeHome/0x01_conversion_project.csv')


head(data)


str(data)


summary(data)


sort(unique(data$age), decreasing=TRUE)


subset(data, age>79)


data = subset(data, age<80)


data_country = data get_ipython().run_line_magic(">%", "")
    group_by(country) get_ipython().run_line_magic(">%", "")
    summarise(conversion_rate = mean(converted))

ggplot(data=data_country, aes(x=country, y=conversion_rate)) +
    geom_bar(stat='identity', aes(fill=country))


data_pages = data get_ipython().run_line_magic(">%", "")
    group_by(total_pages_visited) get_ipython().run_line_magic(">%", "")
    summarise(conversion_rate = mean(converted))
qplot(total_pages_visited, conversion_rate, data=data_pages, geom='line')


data$converted = as.factor(data$converted)  # let's make the class a factor
data$new_user = as.factor(data$new_user)  # also this a factor
levels(data$country)[levels(data$country)=="Germany"]="DE" # Shorter name, easier to plot.


train_sample = sample(nrow(data), size = nrow(data)*0.66) 
train_data = data[train_sample,] 
test_data = data[-train_sample,] 


rf = randomForest(
    y = train_data$converted, 
    x = train_data[, -ncol(train_data)], 
    ytest = test_data$converted, 
    xtest = test_data[, -ncol(test_data)], 
    ntree = 100, 
    mtry = 3, 
    keep.forest = TRUE) 


varImpPlot(rf, type=2)


rf = randomForest(
    y = train_data$converted, 
    x = train_data[, -c(5, ncol(train_data))],  # 5: total page visited
    ytest = test_data$converted, 
    xtest = test_data[, -c(5, ncol(train_data))], 
    ntree = 100, 
    mtry = 3, 
    keep.forest = TRUE,
    classwt = c(0.7,0.3))


varImpPlot(rf, type=2)


# op <- par(mfrow=c(2, 2)) 
# partialPlot(rf, train_data, country, 1) 
# partialPlot(rf, train_data, age, 1) 
# partialPlot(rf, train_data, new_user, 1) 
# partialPlot(rf, train_data, source, 1)


tree = rpart(
    data$converted ~ ., 
    data[, -c(5, ncol(data))], 
    control = rpart.control(maxdepth = 3), 
    parms = list(prior = c(0.7, 0.3))) 
tree
