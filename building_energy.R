##############################################################
##
## Summary: In this notebook I have leveraged the research paper published Athanasios Tsanas & Angeliki Xifara 
##          to develop a machine learning model which could take into consideration 8 different parameters for
##          a building and determine the heating and cooling load. 
##
##          These parameters are:
##
##            Relative Compactness
##            Surface Area
##            Wall Area
##            Roof Area
##            Overall Height
##            Orientation 
##            Glazing area
##            Glazing area distribution
##
##          Research paper available at https://www.sciencedirect.com/science/article/abs/pii/S037877881200151X
##
##
## Areas Covered: In this notebook I have focussed on two main aspects
##          
##            EDA             : Understand the correlation across these features
##            Data Processing : Split data into training and test dataset in a 80:20 ration
##            Model           : Use Random Forest algorithm to train a model which maps the input to output features
##            Tuning          : Run a grid search to identify the optimal hyperparameters for the Random Forest models
##            Prediction      : Check the performance of the model on the test
##
##  Technical Aspect: I have leverage h2o which is an open source library for machine learning. It's available as 
##                    package in R.

#install h2o package
#install.packages("h2o")

#load h2o package
library(h2o)

#initialize h2o which will start a JVM process and return a reference to it
localH2o <- h2o.init(nthreads = -1)

#set the current working directory
setwd('C:\\MachineLearning\\repos\\harvardx\\Data-Science-With-R\\R-BuildingEnergy Model')

#load the input data csv file as R dataframe
data.r <- read.csv('ENB2012_data.csv')

#convert R dataframe to H2o frame
data <- as.h2o(data.r)

#see top entries in the dataframe
head(data)

#see dimensions of the dataframe
dim(data)

#convert column X6 abd X8 as features
factorsList <- c("X6", "X8")
data[,factorsList] <- as.factor(data[,factorsList])

#split the dataframe into train and test dataset in a 80:20 ratio
splits <- h2o.splitFrame(data, 0.8)
train <- splits[[1]]
test <- splits[[2]]

########################### EDA ###########################

#summary of the input data
summary(data.r)

#install.packages("corrplot")
library('corrplot')

corresult<- cor(data.r)

#Positive correlations are displayed in blue and negative correlations in red color. Color intensity and the 
#size of the circle are proportional to the correlation coefficients. In the right side of the correlogram, 
#the legend color shows the correlation coefficients and the corresponding colors.
corrplot(corresult, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

#install.packages("PerformanceAnalytics")
library("PerformanceAnalytics")

#In the above plot:

#  The distribution of each variable is shown on the diagonal.
#  On the bottom of the diagonal : the bivariate scatter plots with a fitted line are displayed
#  On the top of the diagonal : the value of the correlation plus the significance level as stars
#  Each significance level is associated to a symbol : p-values(0, 0.001, 0.01, 0.05, 0.1, 1) <=> symbols("***", "**", "*", ".", " ")
chart.Correlation(data.r, histogram=TRUE, pch=19)

# Get some colors
col<- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = corresult, col = col, symm = TRUE)
########################### EDA ###########################


############################ Model 1 #############################
#
# This model will try and predict the 'Cooling Load' feature 
#
##################################################################
x <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")
y <- "Y2"

#setup the grid to do hyperparameter tuning for a RandomForest model
g <- h2o.grid("randomForest",
              hyper_params = list(
                ntrees = c(50, 100, 120),
                max_depth = c(40, 60),
                min_rows = c(1, 2)
              ),
              x = x, y = y, training_frame = train, nfolds = 10
)

#evaluate the outcome of the grid
g_rmse = h2o.getGrid(g@grid_id, sort_by = "rmse")
as.data.frame( h2o.getGrid(g@grid_id, sort_by = "rmse")@summary_table )

#get the best model
best_model <- h2o.getModel(g_rmse@model_ids[[1]])

#print RMSE of the best model
h2o.rmse(best_model)

#predict on test dataset
perf <- h2o.performance(best_model, test)
perf

rmse_results <- data.frame(method = "Cooling load model (Test Accuracy)", RMSE = h2o.rmse(perf))

############################ Model 2 #############################
#
# This model will try and predict the 'Heating Load' feature 
#
##################################################################
x <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")
y <- "Y1"

#setup the grid to do hyperparameter tuning for a RandomForest model
g <- h2o.grid("randomForest",
              hyper_params = list(
                ntrees = c(50, 100, 120),
                max_depth = c(40, 60),
                min_rows = c(1, 2)
              ),
              x = x, y = y, training_frame = train, nfolds = 10
)

#evaluate the outcome of the grid
g_rmse = h2o.getGrid(g@grid_id, sort_by = "rmse")
as.data.frame( h2o.getGrid(g@grid_id, sort_by = "rmse")@summary_table )

#get the best model
best_model <- h2o.getModel(g_rmse@model_ids[[1]])

#print RMSE of the best model
h2o.rmse(best_model)

#predict on test dataset
perf <- h2o.performance(best_model, test)
perf

rmse_results <- rbind(rmse_results,
                          data.frame(method = "Heating load model (Test Accuracy)", RMSE = h2o.rmse(perf))
                          )

  ################################### Summary of Models ###########################################
rmse_results
